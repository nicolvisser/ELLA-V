from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from torch.utils.data import DataLoader

from ellav.tokenizer import ELLAVTokenizer
from ellav.transformer.config import TransformerModelArgs
from ellav.utils.data import collate_fn
from ellav.utils.dataset import ELLAVTokenizedDataset
from ellav.utils.trainer import ELLAVGARLightningModel, ELLAVTrainArgs

k = 500
lmbda = 0
batch_size = 8
num_workers = 23
accumulate_grad_batches = 8


model_args = TransformerModelArgs(
    dim=1024,
    n_layers=12,
    hidden_dim=4096,
    head_dim=64,
    n_heads=16,
    n_kv_heads=16,
    norm_eps=1e-6,
    vocab_size=None,
    rope_theta=1e4,
)


# Units (1-of-k values between 0 and k-1) are stored as strings in .txt files. E.g.
# u74 u74 u74 u123 u123 u46 u46 u249 u249 u249 ...
# the units are repeated (duplicated) at 50 Hz to indicate their duration.
tokenizer = ELLAVTokenizer.from_phone_vocab(
    phone_vocab={f"u{i}": i for i in range(k)},
    phone_token_rate=50,  # 50 Hz for WavLM units
    num_codec_types=4096,  # WavTokenizer_small_600_24k_4096 has 4096 types
    codec_token_rate=40,  # WavTokenizer_small_600_24k_4096 has 40 Hz tokens
    tokens_not_to_sandwich=[],  # use all
)

model_args.vocab_size = tokenizer.vocab_size

train_args = ELLAVTrainArgs(
    project_name="ella-v-units",
    run_name=f"wavlm-layer-11-km-{k}-lmbda-{lmbda}-train-clean-10k-40hz-test",
    phones_train_dir=f"/mnt/wsl/nvme/code/wavlm-quantize-abx/output/layer-11/str/km-{k}-lmbda-{lmbda}/train",
    phones_val_dir=f"/mnt/wsl/nvme/code/wavlm-quantize-abx/output/layer-11/str/km-{k}-lmbda-{lmbda}/dev",
    codec_train_dir="/mnt/wsl/nvme/code/ella-v-data/WavTokenizer_small_600_24k_4096/train",
    codec_val_dir="/mnt/wsl/nvme/code/ella-v-data/WavTokenizer_small_600_24k_4096/dev",
    batch_size=batch_size,
    num_workers=num_workers,
    lr_init=1e-7,
    warmup_steps=1000,
    lr_max=5e-4,
    decay_steps=9000,
    lr_final=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-9,
    accelerator="gpu",
    strategy="auto",
    devices=1,
    precision="bf16-mixed",
    fast_dev_run=False,
    max_steps=10000,
    val_check_interval=0.5,
    check_val_every_n_epoch=1,
    log_every_n_steps=accumulate_grad_batches,
    accumulate_grad_batches=accumulate_grad_batches,
    gradient_clip_algorithm="norm",
    gradient_clip_val=1.0,
    early_stopping_patience=5,
)

torch.set_float32_matmul_precision("medium")

train_dataset = ELLAVTokenizedDataset(
    tokenizer=tokenizer,
    phones_dir=train_args.phones_train_dir,
    codec_dir=train_args.codec_train_dir,
)
val_dataset = ELLAVTokenizedDataset(
    tokenizer=tokenizer,
    phones_dir=train_args.phones_val_dir,
    codec_dir=train_args.codec_val_dir,
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_args.batch_size,
    shuffle=True,
    num_workers=train_args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=False,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=train_args.batch_size,
    shuffle=False,
    num_workers=train_args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=False,
)

model = ELLAVGARLightningModel(model_args, tokenizer, train_args)

logger = pl.loggers.WandbLogger(
    log_model=False,
    project=train_args.project_name,
    name=train_args.run_name,
)

# Get the checkpoint directory path
checkpoint_dir = Path(f"./checkpoints/{train_args.run_name}")

if (checkpoint_dir / "best.ckpt").exists():
    response = input(
        f"Warning: Checkpoint {checkpoint_dir / 'best.ckpt'} already exists. Continue and overwrite? (y/n): "
    )
    if response.lower() != "y":
        print("Training aborted.")
        exit()

checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Save the tokenizer, model args and trainer args
tokenizer.save_json(checkpoint_dir / "tokenizer.json", indent=4)
model_args.save_json(checkpoint_dir / "model_args.json", indent=4)
train_args.save_json(checkpoint_dir / "train_args.json", indent=4)

config = {
    "model_args": model_args.to_dict(),
    "train_args": train_args.to_dict(),
}
logger.log_hyperparams(config)

print(f"Checkpoint directory: {checkpoint_dir}")

# Add callbacks for checkpointing and early stopping
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="best",
    monitor="val/loss",
    verbose=True,
    save_last=False,
    save_top_k=1,
    save_weights_only=True,
    mode="min",
)

if train_args.early_stopping_patience is not None:
    early_stopping_callback = EarlyStopping(
        monitor="val/loss",
        patience=train_args.early_stopping_patience,
        mode="min",
        verbose=True,
    )
else:
    early_stopping_callback = None

lr_monitor_callback = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
    logger=logger,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        lr_monitor_callback,
    ],
    accelerator=train_args.accelerator,
    strategy=train_args.strategy,
    devices=train_args.devices,
    precision=train_args.precision,
    fast_dev_run=train_args.fast_dev_run,
    max_steps=train_args.max_steps,
    val_check_interval=train_args.val_check_interval,
    check_val_every_n_epoch=train_args.check_val_every_n_epoch,
    log_every_n_steps=train_args.log_every_n_steps,
    accumulate_grad_batches=train_args.accumulate_grad_batches,
    gradient_clip_algorithm=train_args.gradient_clip_algorithm,
    gradient_clip_val=train_args.gradient_clip_val,
)
trainer.fit(model, train_loader, val_loader)
