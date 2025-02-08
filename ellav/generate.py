from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from .transformer.cache import BufferCache
from .transformer.inference import TransformerModel


@torch.inference_mode()
def generate(
    encoded_prompts: List[List[int]],
    model: TransformerModel,
    max_tokens: int,
    max_codec_tokens_per_phone: int,
    temperature: float,
    top_p: float,
    eop_id: int,
    eos_id: int,
    chunk_size: Optional[int] = None,
    progress: bool = False,
) -> Tuple[List[List[int]], List[bool]]:
    model = model.eval()
    B, V = len(encoded_prompts), model.args.vocab_size

    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    cache = BufferCache(
        n_layers=model.args.n_layers,
        max_batch_size=len(encoded_prompts),
        max_seq_len=cache_window,
        n_kv_heads=model.args.n_kv_heads,
        head_dim=model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # Bookkeeping
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s : s + chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache,
        )

        last_token_prelogits = prelogits.index_select(
            0,
            torch.tensor(
                [len(p) for p in prompt_chunks], device=prelogits.device
            ).cumsum(dim=0)
            - 1,
        )
        assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tensors = []
    num_phones = torch.tensor([len(p) - 2 for p in encoded_prompts])
    num_tokens_since_last_eop = torch.tensor([0 for _ in range(B)])
    current_phone_idx = torch.tensor([0 for _ in range(B)])

    if progress:
        pbar = tqdm(
            range(max_tokens),
            desc="ELLA-V: Generating...",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} tokens",
        )
    else:
        pbar = range(max_tokens)

    assert last_token_prelogits is not None
    for _ in pbar:
        next_token_id = sample(
            last_token_prelogits, temperature=temperature, top_p=top_p
        )

        reached_eop = (next_token_id == eop_id).cpu()
        num_tokens_since_last_eop += 1
        num_tokens_since_last_eop[reached_eop] = 0
        force_eop = num_tokens_since_last_eop > (max_codec_tokens_per_phone + 1)
        current_phone_idx[reached_eop] += 1
        force_eos = current_phone_idx >= num_phones
        force_phn = reached_eop & ~force_eos

        if force_eos.all():
            break

        generated_tensors.append(next_token_id[:, None])
        last_token_prelogits = model.forward(
            next_token_id, seqlens=[1] * B, cache=cache
        )

        force_token = force_phn | force_eop | force_eos

        if force_token.any():
            # condition
            force_token = (
                force_token.to(last_token_prelogits.device).unsqueeze(1).expand(B, V)
            )  # Shape: (B, V)

            # input1: create forced logits
            forced_logits = torch.zeros_like(last_token_prelogits)  # (B, V)
            for i in range(B):
                if force_phn[i]:
                    next_token_id = encoded_prompts[i][current_phone_idx[i]]
                    forced_logits[i, next_token_id] = 1e9
                if force_eop[i]:
                    forced_logits[i, eop_id] = 1e9
                if force_eos[i]:
                    forced_logits[i, eos_id] = 1e9

            # Blend original logits with forced logits based on force_token mask
            last_token_prelogits = torch.where(
                condition=force_token,
                input=forced_logits,
                other=last_token_prelogits,
            )

        assert last_token_prelogits.shape == (B, V)

    generated_ids: List[List[int]]
    is_finished: List[bool]
    if generated_tensors:
        generated_ids = torch.cat(generated_tensors, 1).tolist()
        is_finished = force_eos.tolist()
    else:
        generated_ids = []
        is_finished = []

    return generated_ids, is_finished


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
