#!/usr/bin/env python3
"""Unified training script for DDSTR.

Supports four modes via the `mode` config:
  - baseline:  Standard PARSeq training
  - CCD:       Confusion-aware Class Decomposition (extended charset from confusion mapping)
  - CLD:       Confidence-based Label Denoising (train on PL LMDB with denoised labels)
  - CLD_CCD:   CLD then CCD (denoised labels + class decomposition)

Modes CCD and CLD_CCD require:
  - unicode_mapping: path to unicode_mapping.json from confusion_and_pl.py
  - pl_root_dir:     root for PL LMDB with decomposed labels

Mode CLD requires:
  - pl_root_dir:     root for PL LMDB with denoised labels (from generate_pl.py)

Mode CLD_CCD requires both CLD's denoised LMDB and CCD's confusion mapping applied on top.
"""
import glob
import json
import math
import os
import random
from pathlib import Path, PurePath

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch

import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.dataset import build_tree_dataset
from strhub.data.module import SceneTextDataModule
from strhub.data.utils import PLCharsetAdapter
from strhub.models.base import BaseSystem


# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    'Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.'
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


def build_pl_charset(base_charset: str, unicode_mapping_path: str) -> tuple[str, dict]:
    """Extend charset_train with Unicode variant characters from unicode_mapping.json."""
    with open(unicode_mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    ext_chars = ''.join(v['unicode'] for v in mapping.values())
    ext_to_base = {v['unicode']: v['base_char'] for v in mapping.values()}
    return base_charset + ext_chars, ext_to_base


class ValPredictionLogger(Callback):
    """Logs random pred/gt pairs with images as a wandb table each validation."""

    def __init__(self):
        super().__init__()
        self.val_data = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        images, labels = batch
        with torch.no_grad():
            logits = pl_module(images)
        probs = logits.softmax(-1)
        preds, _ = pl_module.tokenizer.decode(probs)
        for img, pred_raw, gt in zip(images, preds, labels):
            pred_mapped = pl_module.charset_adapter(pred_raw)
            img = (img.cpu() * 0.5 + 0.5).clamp(0, 1)
            self.val_data.append((img, gt, pred_raw, pred_mapped))

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.val_data or not trainer.logger:
            return
        samples = random.sample(self.val_data, min(5, len(self.val_data)))
        table = wandb.Table(columns=['image', 'gt', 'pred', 'pred_raw', 'correct'])
        for img, gt, pred_raw, pred_mapped in samples:
            table.add_data(wandb.Image(img), gt, pred_mapped, pred_raw, gt == pred_mapped)
        trainer.logger.experiment.log({'val_samples': table}, step=trainer.global_step)
        self.val_data.clear()


class PLSceneTextDataModule(SceneTextDataModule):
    """Data module that uses PL LMDB for training and original LMDB for validation."""

    def __init__(self, pl_root_dir: str, use_pl_data: bool = True,
                 normalize_unicode_train: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.pl_root_dir = pl_root_dir
        self.use_pl_data = use_pl_data
        self.normalize_unicode_train = normalize_unicode_train

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            transform = self.get_transform(self.img_size, self.augment)
            if self.use_pl_data:
                root = PurePath(self.pl_root_dir, 'train', self.train_dir)
                self._train_dataset = build_tree_dataset(
                    root, self.charset_train, self.max_label_length,
                    self.min_image_dim, self.remove_whitespace,
                    self.normalize_unicode_train,
                    transform=transform,
                )
            else:
                root = PurePath(self.root_dir, 'train', self.train_dir)
                self._train_dataset = build_tree_dataset(
                    root, self.charset_train, self.max_label_length,
                    self.min_image_dim, self.remove_whitespace,
                    self.normalize_unicode, transform=transform,
                )
        return self._train_dataset


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    project_root = Path(hydra.utils.get_original_cwd())

    def resolve_path(p):
        p = Path(p).expanduser()
        if not p.is_absolute():
            p = project_root / p
        return str(p)

    # Determine mode
    mode = config.get('mode', 'baseline')  # baseline | CCD | CLD | CLD_CCD
    use_decomposition = mode in ('CCD', 'CLD_CCD')  # modes that use extended charset
    use_denoised_data = mode in ('CLD', 'CLD_CCD')  # modes that use PL LMDB

    with open_dict(config):
        config.data.root_dir = resolve_path(config.data.root_dir)

        # PL root dir (for CLD, CLD_CCD modes: denoised LMDB; for CCD: decomposed LMDB)
        if not config.get('pl_root_dir'):
            config.pl_root_dir = config.data.root_dir + '/decomposed'
        else:
            config.pl_root_dir = resolve_path(config.pl_root_dir)

        # Unicode mapping for A, BA modes
        if not config.get('unicode_mapping'):
            config.unicode_mapping = 'confusion_pl_output/unicode_mapping.json'
        config.unicode_mapping = resolve_path(config.unicode_mapping)

        # GPU handling
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'
        if gpu and devices > 1:
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    # Extend charset for decomposition modes (CCD, CLD_CCD)
    ext_to_base = {}
    if use_decomposition:
        pl_charset, ext_to_base = build_pl_charset(config.model.charset_train, config.unicode_mapping)
        print(f'[Mode {mode}] Base charset: {len(config.model.charset_train)} -> Extended: {len(pl_charset)}')
        with open_dict(config):
            config.model.charset_train = pl_charset

    model: BaseSystem = hydra.utils.instantiate(config.model)

    # For decomposition modes, use PLCharsetAdapter for eval
    if use_decomposition:
        model.charset_adapter = PLCharsetAdapter(config.data.charset_test, ext_to_base)

    print(summarize(model, max_depth=2))

    # Data module
    use_pl_data = use_denoised_data or use_decomposition
    if use_pl_data:
        # For CCD mode with decomposition: normalize_unicode=False to preserve extended chars
        # For CLD mode with denoising: normalize_unicode=True (labels are normal text)
        normalize_unicode_train = not use_decomposition
        datamodule = PLSceneTextDataModule(
            pl_root_dir=config.pl_root_dir,
            use_pl_data=True,
            normalize_unicode_train=normalize_unicode_train,
            root_dir=config.data.root_dir,
            train_dir=config.data.train_dir,
            img_size=config.data.img_size,
            max_label_length=config.data.max_label_length,
            charset_train=config.model.charset_train,
            charset_test=config.data.charset_test,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            augment=config.data.augment,
            remove_whitespace=config.data.get('remove_whitespace', True),
            normalize_unicode=config.data.get('normalize_unicode', True),
        )
    else:
        datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    _train_root = (
        str(PurePath(config.pl_root_dir, 'train', config.data.train_dir))
        if use_pl_data
        else str(PurePath(config.data.root_dir, 'train', config.data.train_dir))
    )
    print(f'[Mode {mode}] Train data: {_train_root}')
    print(f'[Mode {mode}] Val data:   {PurePath(config.data.root_dir, "val")}')

    cwd = (
        HydraConfig.get().runtime.output_dir
        if config.ckpt_path is None
        else str(Path(config.ckpt_path).parents[1].absolute())
    )
    checkpoint = ModelCheckpoint(
        monitor='val_accuracy', mode='max', save_top_k=3, save_last=True,
        filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
        dirpath=cwd + '/checkpoints',
    )
    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)

    def list_lmdbs(root: str) -> list[str]:
        return sorted(
            str(Path(p).parent.relative_to(root))
            for p in glob.glob(str(Path(root) / '**/data.mdb'), recursive=True)
        )

    overrides = HydraConfig.get().overrides.task
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        group=config.wandb.group,
        name=config.wandb.name,
        save_dir=cwd,
        log_model=False,
    )
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        wandb_logger.experiment.config.update({
            'mode': mode,
            'overrides': list(overrides),
            'dataset/train_root': _train_root,
            'dataset/train_lmdbs': list_lmdbs(_train_root),
        })

    callbacks = [checkpoint, swa, ValPredictionLogger()]
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=wandb_logger,
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
