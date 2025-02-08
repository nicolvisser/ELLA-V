"""
Stripped down version of some of the earlier releases of Mistral from https://github.com/mistralai/mistral-inference and https://github.com/mistralai/mistral-finetune
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from xformers.ops.fmha import memory_efficient_attention

from .cache import BufferCache, CacheInputMetadata, CacheView
from .config import TransformerModelArgs
from .rope import apply_rotary_emb, precompute_freqs_cis


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class Attention(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = args.n_heads // args.n_kv_heads

        self.scale = args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(
            xq, key, val, None if cache is None else cache.mask
        )
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)

        self.feed_forward = FeedForward(args=args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class TransformerModel(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.args = args

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = torch.nn.Linear(args.dim, self.args.vocab_size, bias=True)
        # I accidentally set bias=True here before training, which is not what the original Mistral code does
        # You can try setting bias=False if you want

        # set lazily
        self._freqs_cis = None

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's  dtype means we cannot register it as a buffer
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 8_000, theta=self.args.rope_theta
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)
        return self._freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[BufferCache] = None,
    ) -> torch.Tensor:
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        (num_toks,) = input_ids.shape
        batch_size = len(seqlens)
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)
        assert batch_size <= cache.max_batch_size, (batch_size, cache.max_batch_size)

        input_metadata: Union[CacheInputMetadata, SimpleInputMetadata]

        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)

        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[input_metadata.positions]

        for layer_id, layer in enumerate(self.layers):
            if cache is not None:
                cache_view = cache.get_view(layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)

        return self.output(self.norm(h)).float()

    @classmethod
    def from_model_dir(
        cls, model_dir: str, map_location: str = "cpu"
    ) -> "TransformerModel":
        model_dir = Path(model_dir)
        checkpoint_path: Path = model_dir / "model.pt"
        model_args_path: Path = model_dir / "model_args.json"
        assert checkpoint_path.exists()
        assert model_args_path.exists()
        model = cls(args=TransformerModelArgs.load_json(model_args_path))
        state_dict = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
        model.load_state_dict(state_dict)
        model.to(device=map_location)
        model.eval()
        return model
        model.eval()
        return model
