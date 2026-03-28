#!/usr/bin/env python3
"""Test script for PARSeq models trained with Pseudo-Labeling (PL) extended charset.

Identical to test.py, but replaces model.charset_adapter with PLCharsetAdapter so that
extended Unicode variant characters predicted by the PL model are mapped back to their
base characters before comparison with ground-truth labels.
"""

import argparse
import json
import string
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import torch

from strhub.data.module import SceneTextDataModule
from strhub.data.utils import PLCharsetAdapter
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    err_sample_count: int = 0     # samples with at least one wrong char (word-level error)
    err_sample_ratio: float = 0.0  # % of samples with error
    err_char_count: int = 0       # total wrong chars (char-level)
    err_char_ratio: float = 0.0   # % of chars that are wrong
    total_chars: int = 0          # total chars (for combined calculation)
    ext_sample_count: int = 0     # samples containing at least one ext char
    ext_sample_ratio: float = 0.0  # % of samples containing at least one ext char
    ext_count: int = 0           # total extended chars predicted
    ext_ratio: float = 0.0       # % of predicted chars that are extended
    ext_correct: float = 0.0     # % of ext chars whose base char matches gt at that position


def print_results_table(results: list[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | #ErrSmp | ErrSmp% | #ErrChr | ErrChr% | #ExtSmp | ExtSmp% | #Ext |  Ext% | ExtAcc% |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|--------:|--------:|--------:|--------:|--------:|-----:|------:|--------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0)
    total_ext_count = 0
    total_ext_correct = 0
    total_pred_chars = 0
    total_ext_samples = 0
    total_err_samples = 0
    total_err_chars = 0
    total_all_chars = 0
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        total_ext_count += res.ext_count
        total_ext_samples += res.ext_sample_count
        total_err_samples += res.err_sample_count
        total_err_chars += res.err_char_count
        total_all_chars += res.total_chars
        # Recover absolute pred chars count for combined ext_ratio
        n_pred_chars = round(res.ext_count / (res.ext_ratio / 100)) if res.ext_ratio > 0 else res.total_chars
        total_pred_chars += n_pred_chars
        total_ext_correct += round(res.ext_correct / 100 * res.ext_count) if res.ext_count > 0 else 0
        print(
            f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} '
            f'| {res.err_sample_count:>7} | {res.err_sample_ratio:>7.2f} '
            f'| {res.err_char_count:>7} | {res.err_char_ratio:>7.2f} '
            f'| {res.ext_sample_count:>7} | {res.ext_sample_ratio:>7.1f} '
            f'| {res.ext_count:>4} | {res.ext_ratio:>5.2f} | {res.ext_correct:>7.1f} |',
            file=file,
        )
    c.accuracy /= c.num_samples
    c.err_sample_count = total_err_samples
    c.err_sample_ratio = 100 * total_err_samples / c.num_samples if c.num_samples > 0 else 0
    c.err_char_count = total_err_chars
    c.total_chars = total_all_chars
    c.err_char_ratio = 100 * total_err_chars / total_all_chars if total_all_chars > 0 else 0
    c.ext_sample_count = total_ext_samples
    c.ext_count = total_ext_count
    c.ext_ratio = 100 * total_ext_count / total_pred_chars if total_pred_chars > 0 else 0
    c.ext_correct = 100 * total_ext_correct / total_ext_count if total_ext_count > 0 else 0
    c.ext_sample_ratio = 100 * total_ext_samples / c.num_samples if c.num_samples > 0 else 0
    print('|-{:-<{w}}-|-----------|----------|---------|---------|---------|---------|---------|---------|------|-------|---------|'.format('----', w=w), file=file)
    print(
        f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} '
        f'| {c.err_sample_count:>7} | {c.err_sample_ratio:>7.2f} '
        f'| {c.err_char_count:>7} | {c.err_char_ratio:>7.2f} '
        f'| {c.ext_sample_count:>7} | {c.ext_sample_ratio:>7.1f} '
        f'| {c.ext_count:>4} | {c.ext_ratio:>5.2f} | {c.ext_correct:>7.1f} |',
        file=file,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='~/data/STR/parseq/english/lmdb')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--unicode_mapping',
        default='confusion_pl_output/unicode_mapping.json',
        help='Path to unicode_mapping.json produced by confusion_and_pl.py',
    )
    args, unknown = parser.parse_known_args()
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    args.unicode_mapping = str(Path(args.unicode_mapping).expanduser().resolve())
    kwargs = parse_model_args(unknown)

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    # Load unicode_mapping.json and build ext_to_base mapping
    with open(args.unicode_mapping, 'r', encoding='utf-8') as f:
        unicode_mapping = json.load(f)
    ext_to_base = {v['unicode']: v['base_char'] for v in unicode_mapping.values()}
    print(f'Loaded unicode mapping: {len(ext_to_base)} extended chars from {args.unicode_mapping}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    # Replace charset_adapter with PLCharsetAdapter to map extended chars back to base chars
    model.charset_adapter = PLCharsetAdapter(charset_test, ext_to_base)

    hp = model.hparams
    datamodule = SceneTextDataModule(
        args.data_root,
        '_unused_',
        hp.img_size,
        hp.max_label_length,
        hp.charset_train,
        charset_test,
        args.batch_size,
        args.num_workers,
        False,
        rotation=args.rotation,
    )

    test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
    if args.new:
        test_set += SceneTextDataModule.TEST_NEW
    test_set = sorted(set(test_set))

    ext_chars = set(ext_to_base.keys())

    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        total_pred_chars = 0
        total_ext_chars = 0
        ext_converted_correct = 0
        ext_converted_total = 0
        samples_with_ext = 0
        err_samples = 0
        err_chars = 0
        total_gt_chars = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct

            # Extended char analysis: run model again to get raw predictions
            logits = model.forward(imgs.to(model.device))
            probs = logits.softmax(-1)
            preds_raw, _ = model.tokenizer.decode(probs)
            for pred_raw, gt in zip(preds_raw, labels):
                total_pred_chars += len(pred_raw)
                # Char-level error: compare converted prediction with GT
                pred_adapted = model.charset_adapter(pred_raw)
                max_len = max(len(pred_adapted), len(gt))
                total_gt_chars += max_len
                sample_has_err = (pred_adapted != gt)
                if sample_has_err:
                    err_samples += 1
                for i in range(max_len):
                    p = pred_adapted[i] if i < len(pred_adapted) else ''
                    g = gt[i] if i < len(gt) else ''
                    if p != g:
                        err_chars += 1
                has_ext = False
                for i, c in enumerate(pred_raw):
                    if c in ext_chars:
                        total_ext_chars += 1
                        has_ext = True
                        base_c = ext_to_base[c]
                        ext_converted_total += 1
                        if i < len(gt) and base_c == gt[i]:
                            ext_converted_correct += 1
                if has_ext:
                    samples_with_ext += 1

        accuracy = 100 * correct / total
        err_sample_ratio = 100 * err_samples / total if total > 0 else 0
        err_char_ratio = 100 * err_chars / total_gt_chars if total_gt_chars > 0 else 0
        ext_ratio = 100 * total_ext_chars / total_pred_chars if total_pred_chars > 0 else 0
        ext_correct_ratio = 100 * ext_converted_correct / ext_converted_total if ext_converted_total > 0 else 0
        ext_sample_ratio = 100 * samples_with_ext / total if total > 0 else 0
        results[name] = Result(name, total, accuracy,
                               err_samples, err_sample_ratio,
                               err_chars, err_char_ratio, total_gt_chars,
                               samples_with_ext, ext_sample_ratio,
                               total_ext_chars, ext_ratio, ext_correct_ratio)

    result_groups = {
        'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB,
        'Benchmark': SceneTextDataModule.TEST_BENCHMARK,
    }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)


if __name__ == '__main__':
    main()
