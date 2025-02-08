"""
Stripped down version of some of the earlier releases of Mistral from https://github.com/mistralai/mistral-inference and https://github.com/mistralai/mistral-finetune
"""

import operator
from functools import reduce
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import AttentionBias, BlockDiagonalCausalMask

from .config import TransformerModelArgs
from .rope import apply_rotary_emb, precompute_freqs_cis


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class Attention(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads
        self.head_dim: int = args.head_dim

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: AttentionBias,
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(seqlen_sum, self.n_heads, self.args.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        key, val = xk, xv

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(xq, key, val, mask)

        return self.wo(output.view(seqlen_sum, -1))


class FeedForward(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
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
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        att_mask: AttentionBias,
    ) -> torch.Tensor:
        r = self.attention(self.attention_norm(x), freqs_cis, att_mask)
        h = x + r

        r = self.feed_forward(self.ffn_norm(h))
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
        self.output = torch.nn.Linear(args.dim, args.vocab_size, bias=True)
        # I accidentally set bias=True here before training, which is not what the original Mistral code does
        # You can try setting bias=False if you want

        # set lazily
        self._freqs_cis = None

    @property
    def dtype(self) -> torch.dtype:
        return self.tok_embeddings.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.tok_embeddings.weight.device

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
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        h = self.tok_embeddings(input_ids)
        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        for layer in self.layers:
            h = layer(h, freqs_cis, att_mask)

        return self.output(self.norm(h)).float()

    def to_model_dir(self, model_dir: str):
        model_dir = Path(model_dir)
        state_dict = self.state_dict()
        torch.save(state_dict, model_dir / "model.pt")
        self.args.save_json(model_dir / "model_args.json", indent=4)

    @classmethod
    def from_model_dir(cls, model_dir: str):
        model_dir = Path(model_dir)
        args = TransformerModelArgs.load_json(model_dir / "model_args.json")
        model = cls(args=args)
        model.load_state_dict(torch.load(model_dir / "model.pt"))
        return model


def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
