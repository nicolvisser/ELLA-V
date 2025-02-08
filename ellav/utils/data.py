"""
Defines the dataflow between dataset, dataloader, tokenizer and model during training.
"""

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ELLAVDatasetItem:
    framewise_phones: List[str]  # (Np,)
    framewise_codec_tokens: torch.Tensor  # (Nc)

    def __post_init__(self):
        assert isinstance(self.framewise_phones, list)
        assert all(isinstance(p, str) for p in self.framewise_phones)
        assert isinstance(self.framewise_codec_tokens, torch.Tensor)
        assert self.framewise_codec_tokens.ndim == 1


@dataclass
class ELLAVTokenizedDatasetItem:
    tokens: List[str]  # (N,)
    ids: torch.Tensor  # (N,)
    user_input_mask: torch.Tensor  # (N,)

    @property
    def src_ids(self) -> torch.Tensor:
        return self.ids[:-1]

    @property
    def tgt_ids(self) -> torch.Tensor:
        return self.ids[1:]

    @property
    def loss_mask(self) -> torch.Tensor:
        return ~self.user_input_mask[1:]

    def __post_init__(self):
        assert isinstance(self.tokens, list)
        assert all(isinstance(t, str) for t in self.tokens)
        assert isinstance(self.ids, torch.Tensor)
        assert isinstance(self.user_input_mask, torch.Tensor)
        assert self.ids.ndim == 1
        assert self.user_input_mask.ndim == 1
        N = len(self.tokens)
        assert self.ids.shape[0] == N
        assert self.user_input_mask.shape[0] == N


@dataclass
class ELLAVTokenizedBatch:
    src_tokens: List[str]  # (N,)
    tgt_tokens: List[str]  # (N,)
    src_ids: torch.Tensor  # (N,)
    tgt_ids: torch.Tensor  # (N,)
    loss_mask: torch.Tensor  # (N,)
    q_seqlen: List[int]  # (B,)

    @property
    def batch_size(self) -> int:
        return len(self.q_seqlen)

    def __post_init__(self):
        # safety checks
        assert isinstance(self.src_tokens, list)
        assert isinstance(self.tgt_tokens, list)
        assert all(isinstance(t, str) for t in self.src_tokens)
        assert all(isinstance(t, str) for t in self.tgt_tokens)
        assert self.src_ids.ndim == 1
        assert self.tgt_ids.ndim == 1
        assert self.loss_mask.ndim == 1
        assert isinstance(self.q_seqlen, list)
        assert all(isinstance(i, int) for i in self.q_seqlen)
        N = sum(self.q_seqlen)
        assert self.src_ids.shape[0] == N
        assert self.tgt_ids.shape[0] == N
        assert self.loss_mask.shape[0] == N

    def to(self, device: torch.device) -> "ELLAVTokenizedBatch":
        return ELLAVTokenizedBatch(
            src_tokens=self.src_tokens,
            tgt_tokens=self.tgt_tokens,
            src_ids=self.src_ids.to(device=device),
            tgt_ids=self.tgt_ids.to(device=device),
            loss_mask=self.loss_mask.to(device=device),
            q_seqlen=self.q_seqlen,
        )

    def cpu(self) -> "ELLAVTokenizedBatch":
        return self.to(device=torch.device("cpu"))

    def cuda(self) -> "ELLAVTokenizedBatch":
        return self.to(device=torch.device("cuda"))


def collate_fn(items: List[ELLAVTokenizedDatasetItem]) -> ELLAVTokenizedBatch:
    src_tokens = [token for item in items for token in item.tokens]
    tgt_tokens = [token for item in items for token in item.tokens]
    src_ids = torch.cat([item.src_ids for item in items])
    tgt_ids = torch.cat([item.tgt_ids for item in items])
    loss_mask = torch.cat([item.loss_mask for item in items])
    seq_len = [item.src_ids.shape[0] for item in items]
    return ELLAVTokenizedBatch(
        src_tokens=src_tokens,
        tgt_tokens=tgt_tokens,
        src_ids=src_ids,
        tgt_ids=tgt_ids,
        loss_mask=loss_mask,
        q_seqlen=seq_len,
    )
