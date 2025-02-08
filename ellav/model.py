from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .generate import generate
from .tokenizer import ELLAVTokenizer
from .transformer.config import TransformerModelArgs as TransformerInferenceModelArgs
from .transformer.inference import TransformerModel as TransformerInferenceModel
from .transformer.training import TransformerModel as TransformerTrainingModel
from .transformer.training import TransformerModelArgs as TransformerTrainingModelArgs


class ELLAVGARModel(nn.Module):
    def __init__(
        self,
        model_args: Union[TransformerTrainingModelArgs, TransformerInferenceModelArgs],
        tokenizer: ELLAVTokenizer,
        train: bool = True,
    ):
        super().__init__()

        self.transformer: Union[TransformerTrainingModel, TransformerInferenceModel]
        if train:
            self.transformer = TransformerTrainingModel(args=model_args)
        else:
            self.transformer = TransformerInferenceModel(args=model_args)

        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
    ) -> torch.Tensor:
        assert isinstance(
            self.transformer, TransformerTrainingModel
        ), "self.transformer must be a training model. Reinitalize with train=True."
        return self.transformer(input_ids, seqlens)

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[List[str]],
        max_tokens: int,
        max_codec_tokens_per_phone: int,
        temperature: float,
        top_p: float,
        chunk_size: Optional[int] = None,
        strict: bool = True,
        progress: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(
            self.transformer, TransformerInferenceModel
        ), "self.transformer must be an inference model. Reinitalize with train=False."

        encoded_prompts = [self.tokenizer.encode_infer(prompt) for prompt in prompts]
        generated_ids, reached_eos = generate(
            encoded_prompts=encoded_prompts,
            model=self.transformer,
            max_tokens=max_tokens,
            max_codec_tokens_per_phone=max_codec_tokens_per_phone,
            temperature=temperature,
            top_p=top_p,
            eop_id=self.tokenizer.EOP_id,
            eos_id=self.tokenizer.EOS_id,
            chunk_size=chunk_size,
            progress=progress,
        )
        decoded_ids = [self.tokenizer.decode(ids) for ids in generated_ids]
        decoded_ids = [torch.tensor(ids) for ids in decoded_ids]
        if strict and not all(reached_eos):
            import warnings

            warnings.warn(
                "Some generated sequences did not reach the EOS token. Consider increasing max_tokens. Otherwise, set strict=False to ignore."
            )
        return decoded_ids, reached_eos

    @classmethod
    def from_pretrained_checkpoint(
        cls, checkpoint_path: str, map_location: Optional[str] = "cpu"
    ) -> "ELLAVGARModel":
        checkpoint = torch.load(
            checkpoint_path, weights_only=True, map_location=map_location
        )
        model_args = TransformerInferenceModelArgs.from_dict(checkpoint["model_args"])
        tokenizer = ELLAVTokenizer.from_dict(checkpoint["tokenizer"])
        model = cls(model_args=model_args, tokenizer=tokenizer, train=False)
        model.transformer.load_state_dict(checkpoint["model_state"])
        model.to(map_location)
        model.eval()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded ELLA-V GAR model with {num_params:,} parameters.")
        return model
