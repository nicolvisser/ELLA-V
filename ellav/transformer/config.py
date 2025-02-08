"""
Stripped down version of some of the earlier releases of Mistral from https://github.com/mistralai/mistral-inference and https://github.com/mistralai/mistral-finetune
"""

from dataclasses import dataclass

from simple_parsing import Serializable


@dataclass
class TransformerModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    max_batch_size: int = 0
