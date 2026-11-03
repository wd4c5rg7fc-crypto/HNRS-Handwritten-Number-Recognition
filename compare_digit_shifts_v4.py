"""Compare CNN digit horizontal shifts across HNRS testcases without changing project files."""
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

MODEL = ROOT / 'models' / 'best_cnn_model.keras'
IMAGES = MEMBER / 'testimages'
OUTPUT = ROOT / 'test_results' / 'digit_shift_comparison_v4.csv'
EXPECTED = {
    1: '12-5+8', 2: '15+23', 3: '8+3*4', 4: '26-3*(17-4)',
    5: '18-9', 6: '36/(3+3)', 7: '24/6', 8: '7*(9-4)',
    9: '7+5', 10: '(8+3)*4', 11: '6*4', 12: '50-27',
    13: '15+(6*3)-4', 14: '100/(5*4)', 15: '4*(16+9)',
    16: '9-(2+3)', 17: '(10+8)/(7-3)', 18: '72/8+6',
    19: '9-(2+3)', 20: '48/6+7*2', 21: '35-18',
    22: '15+3', 23: '2*3',
}


def preprocess(character):
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY) if character.ndim == 3 else character.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    scale = 20.0 / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(binary[y:y + h, x:x + w], (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    moments = cv2.moments(canvas)
    if moments['m00']:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        dx, dy = round(13.5 - cx) + 1, round(13.5 - cy)
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))
    return canvas.astype(np.float32) / 255.0


def recognise_all_shifts(characters, flags, model):
    results = {0: [], -1: [], -2: []}
    for character, from_split in zip(characters, flags):
        symbol = None if from_split else recognise_operator(character)
        if symbol is not None:
            for shift in results:
                results[shift].append(symbol)
            continue
        processed = preprocess(character)
        if processed is None:
            raise ValueError('Empty character detected')
        for shift in results:
            if shift == 0:
                shifted = processed
            else:
                matrix = np.float32([[1, 0, shift], [0, 1, 0]])
                shifted = cv2.warpAffine(processed, matrix, (28, 28),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            probs = model.predict(shifted.reshape(1, 28, 28, 1), verbose=0)[0]
            results[shift].append(str(int(np.argmax(probs))))
    return {shift: ''.join(parts) for shift, parts in results.items()}


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    if not IMAGES.is_dir():
        raise FileNotFoundError(IMAGES)
    paths = sorted((p for p in IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    model = load_model(str(MODEL))
    rows = []
    print(f'Found {len(paths)} testcase images. Testing CNN shifts 0, -1, -2.\n')
    print(f'{"Test":<6}{"Expected":<22}{"Baseline":<22}{"Shift -1":<22}{"Shift -2":<22}')
    print('-' * 95)
    for path in paths:
        number = int(''.join(c for c in path.stem if c.isdigit()))
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot read image')
            characters, boxes = segment_digits(image, method='improved', group_division=True)
            characters, boxes = merge_division_parts(characters, boxes)
            characters, boxes, flags = split_connected_digits(characters, boxes, return_split_flags=True)
            results = recognise_all_shifts(characters, flags, model)
            expected = EXPECTED.get(number)
            rows.append({'testcase': number, 'expected': expected or '',
                         'baseline': results[0], 'shift_minus_1': results[-1],
                         'shift_minus_2': results[-2],
                         'baseline_correct': '' if expected is None else results[0] == expected,
                         'minus_1_correct': '' if expected is None else results[-1] == expected,
                         'minus_2_correct': '' if expected is None else results[-2] == expected,
                         'error': ''})
            print(f'{number:<6}{(expected or "unknown"):<22}{results[0]:<22}{results[-1]:<22}{results[-2]:<22}')
        except Exception as exc:
            print(f'{number:<6}ERROR: {exc}')
            rows.append({'testcase': number, 'expected': EXPECTED.get(number, ''),
                         'baseline': '', 'shift_minus_1': '', 'shift_minus_2': '',
                         'baseline_correct': '', 'minus_1_correct': '', 'minus_2_correct': '',
                         'error': str(exc)})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['testcase', 'expected', 'baseline', 'shift_minus_1',
                                               'shift_minus_2', 'baseline_correct',
                                               'minus_1_correct', 'minus_2_correct', 'error'])
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if r['expected'] and not r['error']]
    print(f'\nEvaluated {len(valid)} labelled testcases:')
    for key, label in [('baseline', 'Baseline'), ('shift_minus_1', 'Shift -1'), ('shift_minus_2', 'Shift -2')]:
        correct = sum(r[key] == r['expected'] for r in valid)
        gains = [r['testcase'] for r in valid if r['baseline'] != r['expected'] and r[key] == r['expected']]
        regressions = [r['testcase'] for r in valid if r['baseline'] == r['expected'] and r[key] != r['expected']]
        print(f'{label:<12}: {correct}/{len(valid)} | Newly correct: {gains} | Regressions: {regressions}')
    print(f'CSV saved to: {OUTPUT}')
    if len(valid) != len(EXPECTED):
        print('WARNING: Not all 23 labelled testcases were evaluated.')


if __name__ == '__main__':
    main()
