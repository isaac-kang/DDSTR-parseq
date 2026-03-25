#!/usr/bin/env python3
"""B Step 1: Extract error information from STR model predictions on training data.

Runs inference on training datasets, collects samples where pred != gt,
and saves a TSV file with info needed for PL selection:
  dataset, sample_idx, pred, gt, pred_mean_lp, gt_mean_lp

dataset  : relative path of the LMDB from data_root/train/{train_dir}/ (e.g. "ArT")
sample_idx: 0-based raw LMDB index (before any label filtering)
"""

import argparse
import csv
import string
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strhub.data.dataset import LmdbDataset
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


class IndexedDataset(Dataset):
    """Wraps LmdbDataset, additionally returning the raw 0-based LMDB index."""

    def __init__(self, base: LmdbDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        raw_idx = self.base.filtered_index_list[i] - 1  # 1-based → 0-based
        return img, label, raw_idx


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='/data/isaackang/data/STR/parseq/english/lmdb')
    parser.add_argument('--train_dir', default='real', help='Subdir under data_root/train/ to scan')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='error_info.tsv', help='Output TSV file')
    args, unknown = parser.parse_known_args()
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    kwargs = parse_model_args(unknown)

    # Full charset so charset_adapter doesn't strip characters during pred decoding
    charset_test = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation
    kwargs.update({'charset_test': charset_test})

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    transform = SceneTextDataModule.get_transform(hp.img_size)

    train_root = Path(args.data_root) / 'train' / args.train_dir
    lmdb_dirs = sorted(p.parent for p in train_root.rglob('data.mdb'))
    if not lmdb_dirs:
        print(f'No LMDBs found under {train_root}')
        return

    errors = []
    for lmdb_dir in lmdb_dirs:
        ds_name = str(lmdb_dir.relative_to(train_root))
        base_ds = LmdbDataset(
            str(lmdb_dir), hp.charset_train, hp.max_label_length,
            normalize_unicode=True, transform=transform,
        )
        if len(base_ds) == 0:
            continue
        dataset = IndexedDataset(base_ds)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        for imgs, labels, raw_indices in tqdm(dataloader, desc=ds_name):
            imgs_device = imgs.to(args.device)

            # Pred logits and probs
            logits = model.forward(imgs_device)
            probs = logits.softmax(-1)
            preds, probs_list = model.tokenizer.decode(probs)

            # Teacher forcing for GT probs
            tgt = model.tokenizer.encode(labels, model.device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            memory = model.model.encode(imgs_device)
            num_steps = tgt_in.shape[1]
            tgt_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=model.device), 1)
            tgt_padding_mask = (tgt_in == model.pad_id) | (tgt_in == model.eos_id)
            out = model.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask)
            gt_logits = model.model.head(out)
            gt_probs_all = gt_logits.softmax(-1)

            for img_idx, (pred, prob, gt) in enumerate(zip(preds, probs_list, labels)):
                pred = model.charset_adapter(pred)
                # Case-insensitive: treat case-only difference as correct
                if pred.lower() != gt.lower():
                    pred_mean_lp = prob.log().mean().item()

                    gt_per_char_probs = []
                    for j in range(len(gt)):
                        token_id = tgt_out[img_idx, j].item()
                        gt_per_char_probs.append(gt_probs_all[img_idx, j, token_id].item())
                    eos_pos = len(gt)
                    gt_eos_prob = gt_probs_all[img_idx, eos_pos, model.eos_id].item()
                    gt_all_p = torch.tensor(gt_per_char_probs + [gt_eos_prob])
                    gt_mean_lp = gt_all_p.log().mean().item()

                    errors.append({
                        'dataset': ds_name,
                        'sample_idx': raw_indices[img_idx].item(),
                        'pred': pred,
                        'gt': gt,
                        'pred_mean_lp': round(pred_mean_lp, 6),
                        'gt_mean_lp': round(gt_mean_lp, 6),
                    })

    # Sort by pred_mean_lp ascending (least confident first)
    errors.sort(key=lambda e: e['pred_mean_lp'])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'sample_idx', 'pred', 'gt', 'pred_mean_lp', 'gt_mean_lp'],
                                delimiter='\t')
        writer.writeheader()
        writer.writerows(errors)

    print(f'Saved {len(errors)} errors to {output_path}')


if __name__ == '__main__':
    main()
