"""Batch-test original labelled testcases with V5.4, MathCNN and baseline-first Hybrid.
Run in HNRS_Combined_Project: python test_hnrs_v64_all23.py
Does not modify models, images or GUI. Writes test_results/v64_all23_comparison.csv.
"""
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

import compare_four_plus_v54 as v54
from HNRS_app_v64_division import (MATH_CLASSES, preprocess_math_character,
                                fuse_math_and_legacy, apply_mathcnn_division_rule)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'test_results' / 'v64_all23_comparison.csv'

def main():
    base = v54.base
    images = base.IMAGES
    if not images.is_dir():
        raise SystemExit(f'Missing test image folder: {images}')
    label_path = ROOT / 'models' / 'class_labels.json'
    math_path = ROOT / 'models' / 'math_cnn_16.keras'
    legacy_path = ROOT / 'models' / 'best_cnn_model.keras'
    for p in (label_path, math_path, legacy_path):
        if not p.is_file():
            raise SystemExit(f'Missing required file: {p}')
    labels = json.loads(label_path.read_text(encoding='utf-8-sig'))
    if labels != MATH_CLASSES:
        raise SystemExit(f'Wrong label order: {labels}')

    legacy_model = load_model(str(legacy_path), compile=False)
    math_model = load_model(str(math_path), compile=False)
    candidates = sorted((p for p in images.glob('testcase*.*')
                         if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                        key=lambda p: int(re.search(r'\d+', p.stem).group())
                        if re.search(r'\d+', p.stem) else 999)
    rows = []
    for path in candidates:
        match = re.search(r'\d+', path.stem)
        number = int(match.group()) if match else -1
        expected = base.EXPECTED.get(number, '')
        row = dict(testcase=path.name, expected=expected, v54='', mathcnn='', hybrid='',
                   v54_correct='', mathcnn_correct='', hybrid_correct='',
                   segments='', notes='', error='')
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Could not read image')
            # Same shared segmentation as GUI Calculator and V5.4 comparisons.
            chars, boxes, flags, parens = v54.v52.v51.v50.segment(image)
            row['segments'] = len(chars)
            old, _ = v54.recognise_v54(chars, boxes, flags, legacy_model, parens)
            batch = []
            for i, char in enumerate(chars):
                processed = preprocess_math_character(char)
                if processed is None:
                    raise ValueError(f'Empty segment {i}')
                batch.append(processed)
            probs = math_model.predict(np.stack(batch)[..., np.newaxis], verbose=0)
            if probs.shape[1] != 16:
                raise ValueError(f'Expected 16 output classes, got {probs.shape}')
            math_symbols = [parens[i] if parens[i] in ('(', ')') else
                            MATH_CLASSES[int(np.argmax(prob))]
                            for i, prob in enumerate(probs)]
            math_symbols, division_notes = apply_mathcnn_division_rule(
                chars, flags, parens, math_symbols
            )
            raw = ''.join(math_symbols)
            fused, notes = fuse_math_and_legacy(math_symbols, probs, old, parens)
            notes = division_notes + notes
            row.update(v54=old, mathcnn=raw, hybrid=fused, notes=' | '.join(notes))
            if expected:
                row.update(v54_correct=(old == expected),
                           mathcnn_correct=(raw == expected),
                           hybrid_correct=(fused == expected))
        except Exception as exc:
            row['error'] = repr(exc)
        rows.append(row)
        print(f'{path.name:18} expected={expected or "UNLABELLED":24} '
              f'V5.4={row["v54"]:24} MathCNN={row["mathcnn"]:24} '
              f'Hybrid={row["hybrid"]:24} {row["error"]}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                ['testcase', 'expected', 'v54', 'mathcnn', 'hybrid',
                                 'v54_correct', 'mathcnn_correct', 'hybrid_correct',
                                 'segments', 'notes', 'error'])
        writer.writeheader()
        writer.writerows(rows)
    labelled = [r for r in rows if r['expected']]
    print('\nSUMMARY (exact transcription, not just valid arithmetic):')
    print(f'Images found: {len(rows)} | Labelled: {len(labelled)}')
    for name in ('v54', 'mathcnn', 'hybrid'):
        correct = sum(r[name] == r['expected'] and not r['error'] for r in labelled)
        print(f'{name:8}: {correct}/{len(labelled)}')
    print('Errors:', [(r['testcase'], r['error']) for r in rows if r['error']])
    print('Results:', OUT)

if __name__ == '__main__':
    main()
