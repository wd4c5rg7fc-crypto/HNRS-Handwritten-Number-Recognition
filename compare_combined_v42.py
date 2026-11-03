"""Read-only V4 baseline vs combined V4.2 (disconnected star + adaptive CNN).

Place beside compare_disconnected_star_v41.py and compare_adaptive_digit_v4.py.
Run: python compare_combined_v42.py
Does not modify app.py, operator_recognition.py, or the CNN model.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
MEMBER = ROOT / 'member_sources' / 'member_2'
sys.path.insert(0, str(MEMBER))

from Segmentation import segment_digits
from division_grouping import merge_division_parts
from connected_digit_splitter import split_connected_digits
from operator_recognition import recognise_operator
from compare_disconnected_star_v41 import candidate_disconnected_star
from compare_adaptive_digit_v4 import preprocess, shifted, EXPECTED

MODEL = ROOT / 'models' / 'best_cnn_model.keras'
IMAGES = MEMBER / 'testimages'
OUTPUT = ROOT / 'test_results' / 'combined_v42_comparison.csv'
THRESHOLD = 0.80


def evaluate(chars, flags, model):
    baseline, combined, changes = [], [], []
    for i, (char, from_split) in enumerate(zip(chars, flags)):
        op = None if from_split else recognise_operator(char)
        if op is not None:
            baseline.append(op)
            combined.append(op)
            continue

        processed = preprocess(char)
        if processed is None:
            raise ValueError(f'Empty character at position {i}')
        p = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)[0]
        original_digit = int(np.argmax(p))
        original = str(original_digit)
        baseline.append(original)

        # V4.1: try operator detection before applying adaptive digit recognition.
        if not from_split and candidate_disconnected_star(char):
            combined.append('*')
            changes.append(f'pos{i}:{original}->* [disconnected star]')
            continue

        best_digit = original_digit
        best_conf = float(p[original_digit])
        best_shift = 0
        if best_conf < THRESHOLD:
            for dx in (-1, -2):
                shifted_img = shifted(processed, dx)
                sp = model.predict(shifted_img.reshape(1, 28, 28, 1), verbose=0)[0]
                digit = int(np.argmax(sp))
                confidence = float(sp[digit])
                if confidence > best_conf:
                    best_digit, best_conf, best_shift = digit, confidence, dx
        combined.append(str(best_digit))
        if best_shift:
            changes.append(f'pos{i}:{original}->{best_digit} [shift {best_shift:+d}]')

    return ''.join(baseline), ''.join(combined), '; '.join(changes)


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    if not IMAGES.is_dir():
        raise FileNotFoundError(IMAGES)
    model = load_model(str(MODEL))
    paths = sorted(
        (p for p in IMAGES.glob('testcase*.*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')),
        key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0)
    )
    print(f'Found {len(paths)} images. Baseline V4 vs combined V4.2 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"Baseline":<23}{"Combined V4.2":<23}Changes')
    print('-' * 105)
    rows = []
    for path in paths:
        number = int(''.join(c for c in path.stem if c.isdigit()))
        row = dict(testcase=number, expected=EXPECTED.get(number, ''),
                   baseline='', combined='', changes='', error='')
        try:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f'Cannot read {path.name}')
            chars, boxes = segment_digits(img, method='improved', group_division=True)
            chars, boxes = merge_division_parts(chars, boxes)
            chars, boxes, flags = split_connected_digits(chars, boxes, return_split_flags=True)
            row['baseline'], row['combined'], row['changes'] = evaluate(chars, flags, model)
            print(f'{number:<6}{(row["expected"] or "unknown"):<22}'
                  f'{row["baseline"]:<23}{row["combined"]:<23}{row["changes"]}')
        except Exception as exc:
            row['error'] = str(exc)
            print(f'{number:<6}ERROR: {exc}')
        rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['testcase', 'expected', 'baseline',
                                               'combined', 'changes', 'error'])
        writer.writeheader()
        writer.writerows(rows)

    labelled = [r for r in rows if r['expected'] and not r['error']]
    baseline_correct = sum(r['baseline'] == r['expected'] for r in labelled)
    combined_correct = sum(r['combined'] == r['expected'] for r in labelled)
    gains = [r['testcase'] for r in labelled
             if r['baseline'] != r['expected'] and r['combined'] == r['expected']]
    regressions = [r['testcase'] for r in labelled
                   if r['baseline'] == r['expected'] and r['combined'] != r['expected']]
    print(f'\nBaseline: {baseline_correct}/{len(labelled)} | Combined V4.2: {combined_correct}/{len(labelled)}')
    print(f'Newly correct: {gains} | Regressions: {regressions}')
    print(f'Testcase4 combined: {next((r["combined"] for r in rows if r["testcase"] == 4), "missing")}')
    print(f'CSV: {OUTPUT}')
    if len(labelled) != len(EXPECTED) or baseline_correct != 15:
        print('WARNING: Baseline/count differs from verified V4 15/23. Investigate before integration.')
    print('No working project files modified.')


if __name__ == '__main__':
    main()
