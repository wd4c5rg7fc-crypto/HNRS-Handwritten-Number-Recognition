"""Compare baseline and +1px horizontal digit preprocessing on HNRS testcases.

Run in project root: python compare_shift_all.py
Does not modify app.py or trained models.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'member_sources' / 'member_2'))
from Segmentation import segment_digits
from expression_evaluator import calculate_expression
from division_grouping import merge_division_parts
from operator_recognition import recognise_operator
from app import preprocess_digit

# Only include expressions whose handwritten ground truth has been confirmed.
# Edit this dictionary if you verify additional originals.
EXPECTED = {
    1: '12-5+8', 2: '15+23', 3: '8+3*4', 4: '26-3*(17-4)',
    5: '18-9', 6: '36/(3+3)', 7: '24/6', 8: '7*(9-4)',
    9: '7+5', 10: '(8+3)*4', 11: '6*4', 12: '50-27',
    13: '15+(6*3)-4', 14: '100/(5*4)', 15: '4*(16+9)',
    16: '9-(2+3)', 17: '(10+8)/(7-3)', 18: '72/8+6',
    19: '9-(2+3)', 20: '48/6+7*2', 21: '35-18',
    22: '15+3', 23: '2*3',
}


def find_images():
    matches = {}
    for folder in [ROOT / 'test_images', ROOT / 'images', ROOT / 'dataset', ROOT]:
        if not folder.exists():
            continue
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                stem = path.stem.lower()
                if stem.startswith('testcase') and stem[8:].isdigit():
                    matches.setdefault(int(stem[8:]), path)
    return sorted(matches.items())


def predict(characters, model, dx):
    symbols = []
    for character in characters:
        operator = recognise_operator(character)
        if operator is not None:
            symbols.append(operator)
            continue
        processed = preprocess_digit(character)
        if processed is None:
            symbols.append('?')
            continue
        if dx:
            transform = np.float32([[1, 0, dx], [0, 1, 0]])
            processed = cv2.warpAffine(processed, transform, (28, 28))
        probabilities = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)[0]
        symbols.append(str(int(np.argmax(probabilities))))
    return ''.join(symbols)


def main():
    images = find_images()
    if not images:
        raise FileNotFoundError('No testcaseN.jpeg/png images found under the project root.')
    model = load_model(str(ROOT / 'models' / 'best_cnn_model.keras'))
    rows = []
    print(f'Found {len(images)} testcase images. Comparing baseline with +1 pixel shift.\n')
    print(f'{"Test":<7} {"Baseline":<19} {"Shift right":<19} {"Base OK":<9} {"Shift OK":<9}')
    print('-' * 73)
    for number, path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f'{number:<7} unreadable: {path}')
            continue
        try:
            chars, boxes = segment_digits(image, method='improved', group_division=True)
            chars, boxes = merge_division_parts(chars, boxes)
            baseline = predict(chars, model, 0)
            shifted = predict(chars, model, 1)
            expected = EXPECTED.get(number)
            base_ok = baseline == expected if expected is not None else None
            shift_ok = shifted == expected if expected is not None else None
            row = {'testcase': number, 'expected': expected or '',
                   'baseline': baseline, 'shift_right': shifted,
                   'baseline_correct': '' if base_ok is None else base_ok,
                   'shift_correct': '' if shift_ok is None else shift_ok,
                   'baseline_answer': str(calculate_expression(baseline)),
                   'shift_answer': str(calculate_expression(shifted))}
            rows.append(row)
            fmt = lambda value: 'unknown' if value is None else ('YES' if value else 'NO')
            print(f'{number:<7} {baseline:<19} {shifted:<19} {fmt(base_ok):<9} {fmt(shift_ok):<9}')
        except Exception as exc:
            print(f'{number:<7} ERROR: {exc}')
    if not rows:
        return
    output = ROOT / 'test_results' / 'shift_comparison.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    known = [r for r in rows if r['expected']]
    baseline_count = sum(r['baseline_correct'] is True for r in known)
    shifted_count = sum(r['shift_correct'] is True for r in known)
    print(f'\nVerified expression accuracy (on {len(known)} labelled images):')
    print(f'  Baseline: {baseline_count}/{len(known)}')
    print(f'  +1 pixel: {shifted_count}/{len(known)}')
    print('  Note: This compares complete expressions, not per-digit accuracy.')
    print(f'CSV: {output}')


if __name__ == '__main__':
    main()
