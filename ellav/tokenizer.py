from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Dict, List

import torch
from simple_parsing import Serializable
from tqdm import tqdm

from .utils.data import ELLAVTokenizedDatasetItem


@dataclass
class ELLAVTokenizer(Serializable):
    token_to_id_dict: Dict[str, int]
    num_phone_types: int
    phone_token_rate: float  # Hz
    num_codec_types: int
    codec_token_rate: float  # Hz
    tokens_not_to_sandwich: List[str] = field(default_factory=lambda: ["UNK", "SIL"])

    num_special_types: int = field(default=4, init=False)
    UNK_token: str = field(default="UNK", init=False)
    BOS_token: str = field(default="BOS", init=False)
    EOP_token: str = field(default="EOP", init=False)
    EOS_token: str = field(default="EOS", init=False)
    UNK_id: int = field(default=0, init=False)
    BOS_id: int = field(default=1, init=False)
    EOP_id: int = field(default=2, init=False)
    EOS_id: int = field(default=3, init=False)

    @property
    def id_to_token_dict(self) -> Dict[str, int]:
        return {v: k for k, v in self.token_to_id_dict.items()}

    @property
    def ids_not_to_sandwich(self) -> List[int]:
        return [self.token_to_id_dict[token] for token in self.tokens_not_to_sandwich]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id_dict)

    @property
    def inference_zero_out_ids(self) -> List[int]:
        """IDs to zero out for inference"""
        zero_out_mask = torch.zeros(len(self.token_to_id_dict), dtype=torch.bool)
        # Set True (to be zeroed out) for UNK and BOS tokens
        zero_out_mask[self.UNK_id] = True
        zero_out_mask[self.BOS_id] = True
        # Set True for all phoine tokens (they come after special tokens)
        phones_start = self.num_special_types
        phones_end = self.num_special_types + self.num_phone_types
        zero_out_mask[phones_start:phones_end] = True
        return zero_out_mask

    @classmethod
    def from_phone_vocab(
        cls,
        phone_vocab: Dict[str, int],
        phone_token_rate: float,
        num_codec_types: int,
        codec_token_rate: float,
        tokens_not_to_sandwich: List[str] = ["UNK", "SIL"],
    ) -> "ELLAVTokenizer":
        num_phone_tokens = len(phone_vocab)

        # special tokens go at the beginning
        special_tokens = {"UNK": 0, "BOS": 1, "EOP": 2, "EOS": 3}
        token_to_id_dict = special_tokens.copy()
        # then the phone tokens
        token_to_id_dict.update(
            {
                phone: idx + len(special_tokens)
                for idx, phone in enumerate(sorted(phone_vocab.keys()))
            }
        )
        # then the codec tokens
        start_idx = len(special_tokens) + num_phone_tokens
        token_to_id_dict.update({str(i): start_idx + i for i in range(num_codec_types)})

        return cls(
            token_to_id_dict=token_to_id_dict,
            num_phone_types=num_phone_tokens,
            phone_token_rate=phone_token_rate,
            num_codec_types=num_codec_types,
            codec_token_rate=codec_token_rate,
            tokens_not_to_sandwich=tokens_not_to_sandwich,
        )

    @classmethod
    def train_from_phone_dir(
        cls,
        phone_dir: str | Path,
        phone_token_rate: float,
        num_codec_types: int,
        codec_token_rate: float,
        delimiter: str = " ",
        tokens_not_to_sandwich: List[str] = ["UNK", "SIL"],
    ) -> "ELLAVTokenizer":
        """Creates a tokenizer from a directory containing files with phone sequences.

        Args:
            phone_dir: Directory containing .txt files with phone sequences
            delimiter: Character used to separate phones in the files

        Returns:
            A new ELLAVTokenizer instance
        """
        assert Path(
            phone_dir
        ).is_dir(), f"Phone directory {phone_dir} does not exist"

        # Get all .txt files in directory and subdirectories
        phone_paths = list(Path(phone_dir).rglob("*.txt"))

        assert len(phone_paths) > 0, f"No phone files found in {phone_dir}"

        # Build initial dictionary of phones
        token_to_id_dict_unsorted = {}
        for path in tqdm(phone_paths, desc="Processing phone files"):
            with open(path, "r") as f:
                text = f.read()
            for phone in text.split(delimiter):
                if phone not in token_to_id_dict_unsorted:
                    token_to_id_dict_unsorted[phone] = len(token_to_id_dict_unsorted)

        return cls.from_phone_vocab(
            phone_vocab=token_to_id_dict_unsorted,
            phone_token_rate=phone_token_rate,
            num_codec_types=num_codec_types,
            codec_token_rate=codec_token_rate,
            tokens_not_to_sandwich=tokens_not_to_sandwich,
        )

    def __post_init__(self):
        # sanity check
        assert len(self.token_to_id_dict) == (
            self.num_special_types + self.num_phone_types + self.num_codec_types
        ), f"Total number of tokens ({len(self.token_to_id_dict)}) must match the sum of special tokens ({self.num_special_types}), conditioning tokens ({self.num_phone_types}), and codec tokens ({self.num_codec_types})"

    def encode_train(
        self,
        framewise_phones: List[str],  # at self.phone_token_rate Hz
        framewise_codec_tokens: torch.Tensor,  # at self.codec_token_rate Hz
        strict: bool = True,  # checks for approximate equal durations
        strict_duration_tolerance: float = 0.1,  # 100ms
    ) -> ELLAVTokenizedDatasetItem:

        assert framewise_codec_tokens.ndim == 1

        if strict:
            # make sure the phones and codec tokens have approximately the same duration given their frequencies
            phones_duration = len(framewise_phones) / self.phone_token_rate
            codec_tokens_duration = len(framewise_codec_tokens) / self.codec_token_rate

            assert (
                abs(phones_duration - codec_tokens_duration)
                <= strict_duration_tolerance
            ), f"The phone and codec tokens should have approximately the same duration of tokens. Got {phones_duration} seconds and {codec_tokens_duration} seconds."

        # framewise phones data: (phone, index)
        framewise_phones_data = [
            (phone, index) for index, phone in enumerate(framewise_phones)
        ]
        framewise_phones_data_grouped = [
            (k, list(g))
            for k, g in groupby(framewise_phones_data, key=lambda x: x[0])
        ]

        # deduplicated phones data: (phone, start_time, priority=1)
        deduped_phones_data = [
            (
                phone,
                list(g)[0][1] / self.phone_token_rate,  # first_idx / freq
                1,  # MEDIUM PRIORITY
            )
            for phone, g in framewise_phones_data_grouped
            if phone not in self.tokens_not_to_sandwich
        ]

        # framewised codec data: (codec_id, start_time, priority=2)
        framewise_codec_data = [
            (
                str(codec_id.item()),
                index / self.codec_token_rate,  # index / freq
                2,  # LOWEST PRIORITY
            )
            for index, codec_id in enumerate(framewise_codec_tokens)
        ]

        # eop data: (eop_id, start_time, priority=0)
        eop_data = [
            (
                self.EOP_token,
                (list(g)[-1][1] + 1) / self.phone_token_rate,  # (last_idx + 1) / freq
                0,  # HIGHEST PRIORITY
            )
            for phone, g in framewise_phones_data_grouped
            if phone not in self.tokens_not_to_sandwich
        ]

        # global advance tokens
        global_advance_tokens = [item[0] for item in deduped_phones_data]

        # interleaved data
        interleaved_data = sorted(
            eop_data + deduped_phones_data + framewise_codec_data,
            key=lambda x: (x[1], x[2]),  # sort first by timestamp, then by priority
        )

        # interleaved tokens
        interleaved_tokens = [item[0] for item in interleaved_data]

        hybrid_tokens = (
            global_advance_tokens
            + [self.BOS_token]
            + interleaved_tokens
            + [self.EOS_token]
        )

        # Update tokens to UNK if not in dictionary and get their IDs
        hybrid_tokens = [
            token if token in self.token_to_id_dict else self.UNK_token
            for token in hybrid_tokens
        ]
        hybrid_ids = torch.tensor(
            [self.token_to_id_dict[token] for token in hybrid_tokens]
        )

        user_input_mask = torch.logical_or(
            torch.logical_and(
                hybrid_ids > self.num_special_types,
                hybrid_ids < self.num_special_types + self.num_phone_types,
            ),
            hybrid_ids == self.BOS_id,
        )

        return ELLAVTokenizedDatasetItem(
            tokens=hybrid_tokens,
            ids=hybrid_ids,
            user_input_mask=user_input_mask,
        )

    def encode_infer(self, phones: List[str]) -> List[int]:
        # make sure the phones are deduplicated
        phones = [
            k for k, _ in groupby(phones) if k not in self.tokens_not_to_sandwich
        ]
        # add the BOS token and the first phone
        tokens = phones + [self.BOS_token, phones[0]]

        # encode to ids
        ids = [self.token_to_id_dict.get(token, self.UNK_id) for token in tokens]
        return ids

    def decode(self, ids: List[int]) -> List[int]:
        """Decode a sequence of hybrid ids into a stream of acoustic (codec)tokens"""
        # return only the tokens corresponding to the codec tokens
        ids = [
            id for id in ids if id >= self.num_special_types + self.num_phone_types
        ]
        tokens = [self.id_to_token_dict[id] for id in ids]
        codec_ids = [int(token) for token in tokens]
        return codec_ids
