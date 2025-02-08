"""
Contains the Lightning Module used to train the ELLA-V model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import lightning as L
import torch
from simple_parsing import Serializable

from ..tokenizer import ELLAVTokenizer
from ..transformer.config import TransformerModelArgs
from ..transformer.training import TransformerModel
from .data import ELLAVTokenizedBatch
from .scheduler import LinearRampCosineDecayScheduler


@dataclass
class ELLAVTrainArgs(Serializable):
    # -------------------------------------- wandb -------------------------------------
    project_name: str
    run_name: str
    # ----------------------------------- dataloader -----------------------------------
    phones_train_dir: str
    phones_val_dir: str
    codec_train_dir: str
    codec_val_dir: str
    batch_size: int
    num_workers: int
    # ------------------------------------ optimizer -----------------------------------
    lr_init: float
    warmup_steps: int
    lr_max: float
    decay_steps: int
    lr_final: float
    betas: Tuple[float, float]
    weight_decay: float
    eps: float
    # ----------------------------------- pl.trainer -----------------------------------
    accelerator: str
    strategy: str
    devices: int
    precision: str
    fast_dev_run: bool
    max_steps: int
    val_check_interval: float
    check_val_every_n_epoch: int
    log_every_n_steps: int
    accumulate_grad_batches: int
    gradient_clip_algorithm: Optional[str]
    gradient_clip_val: Optional[float]
    early_stopping_patience: Optional[int]


class ELLAVGARLightningModel(L.LightningModule):
    """Lightning module to train ELLAV GAR model"""

    def __init__(
        self,
        model_args: TransformerModelArgs,
        tokenizer: ELLAVTokenizer,
        train_args: ELLAVTrainArgs,
    ):
        super().__init__()
        self.model_args = model_args
        self.tokenizer = tokenizer
        self.train_args = train_args
        self.model = TransformerModel(args=model_args)

    def forward(self, batch: ELLAVTokenizedBatch) -> torch.Tensor:
        logits = self.model.forward(input_ids=batch.src_ids, seqlens=batch.q_seqlen)
        logits_for_loss = logits[batch.loss_mask]
        tgt_ids_for_loss = batch.tgt_ids[batch.loss_mask]
        loss = torch.nn.functional.cross_entropy(
            input=logits_for_loss,
            target=tgt_ids_for_loss,
            ignore_index=self.tokenizer.UNK_id,
        )
        return loss

    def training_step(self, batch: ELLAVTokenizedBatch, batch_idx: int) -> torch.Tensor:
        loss = self(batch=batch)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            batch_size=batch.batch_size,
            sync_dist=True,
        )

        # tokens_per_batch = batch.src_ids.shape[0]
        # self.log(
        #     "train/tokens_per_batch",
        #     tokens_per_batch, prog_bar=True,
        #     batch_size=batch.batch_size,
        #     sync_dist=True, reduce_fx="sum"
        # )
        return loss

    def validation_step(self, batch: ELLAVTokenizedBatch, batch_idx: int) -> None:
        loss = self(batch=batch)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            batch_size=batch.batch_size,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_args.lr_max,
            betas=self.train_args.betas,
            weight_decay=self.train_args.weight_decay,
            eps=self.train_args.eps,
        )

        sched_config = {
            "scheduler": LinearRampCosineDecayScheduler(
                optimizer,
                n_linear_steps=self.train_args.warmup_steps,
                n_decay_steps=self.train_args.decay_steps,
                lr_init=self.train_args.lr_init,
                lr_max=self.train_args.lr_max,
                lr_final=self.train_args.lr_final,
            ),
            "frequency": 1,
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": sched_config}

    @classmethod
    def from_model_dir(cls, model_dir: str):
        model_dir = Path(model_dir)
        tokenizer_path: Path = model_dir / "tokenizer.json"
        checkpoint_path: Path = model_dir / "best.ckpt"
        model_args_path: Path = model_dir / "model_args.json"
        train_args_path: Path = model_dir / "train_args.json"
        assert tokenizer_path.exists(), tokenizer_path
        assert checkpoint_path.exists(), checkpoint_path
        assert model_args_path.exists(), model_args_path
        assert train_args_path.exists(), train_args_path
        return cls.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_args=TransformerModelArgs.load(model_args_path),
            tokenizer=ELLAVTokenizer.load_json(tokenizer_path),
            train_args=ELLAVTrainArgs.load(train_args_path),
            strict=False,
        )
