"""
Dataset used during training.
"""

from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset

from ..tokenizer import ELLAVTokenizer
from .data import ELLAVDatasetItem, ELLAVTokenizedDatasetItem


class ELLAVDataset(Dataset):
    def __init__(
        self,
        phone_strings_dir: Union[str, Path],
        codec_tokens_dir: Union[str, Path],
        delimiter: str = " ",
    ):
        self.phone_tokens_dir = Path(phone_strings_dir)
        self.codec_tokens_dir = Path(codec_tokens_dir)
        self.delimiter = delimiter

        self.phone_strings_paths = {
            path.stem: path for path in self.phone_tokens_dir.rglob("*.txt")
        }
        self.codec_tokens_paths = {
            path.stem: path for path in self.codec_tokens_dir.rglob("*.pt")
        }

        print(f"Found {len(self.phone_strings_paths)} phone token files")
        print(f"Found {len(self.codec_tokens_paths)} codec token files")

        # Find common keys
        common_keys = set(self.phone_strings_paths.keys()) & set(
            self.codec_tokens_paths.keys()
        )
        print(f"Found {len(common_keys)} common keys")

        # Filter paths to keep only common keys
        self.phone_strings_paths = {
            k: v for k, v in self.phone_strings_paths.items() if k in common_keys
        }
        self.codec_tokens_paths = {
            k: v for k, v in self.codec_tokens_paths.items() if k in common_keys
        }

        self.keys = sorted(
            list(common_keys)
        )  # Use common_keys directly instead of phone_strings_paths keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int) -> ELLAVDatasetItem:
        key = self.keys[index]
        phones = (
            Path(self.phone_strings_paths[key]).read_text().split(self.delimiter)
        )

        codec_tokens = torch.load(
            self.codec_tokens_paths[key],
            weights_only=True,
        )

        return ELLAVDatasetItem(
            framewise_phones=phones,
            framewise_codec_tokens=codec_tokens,
        )


class ELLAVTokenizedDataset(Dataset):
    def __init__(
        self,
        tokenizer: ELLAVTokenizer,
        phones_dir: Union[str, Path],
        codec_dir: Union[str, Path],
    ):
        self.ellav_dataset = ELLAVDataset(
            phone_strings_dir=phones_dir,
            codec_tokens_dir=codec_dir,
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ellav_dataset)

    def __getitem__(self, index: int) -> ELLAVTokenizedDatasetItem:
        item = self.ellav_dataset[index]
        return self.tokenizer.encode_train(
            framewise_phones=item.framewise_phones,
            framewise_codec_tokens=item.framewise_codec_tokens,
        )
