"""Read-only V4.2 vs V4.3 experiment: parentheses masks + 2-component star.
Place in HNRS_Combined_Project beside parenthesis_mask_splitter_v5.py and
compare_adaptive_digit_v4.py. Run: python compare_testcase10_v43.py
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
from operator_recognition import recognise_operator, _binary_crop, _asterisk
from parenthesis_mask_splitter_v5 import isolate_parentheses
from compare_adaptive_digit_v4 import preprocess, shifted, EXPECTED

MODEL = ROOT / 'models' / 'best_cnn_model.keras'
IMAGES = MEMBER / 'testimages'
OUTPUT = ROOT / 'test_results' / 'testcase10_v43_comparison.csv'


def star_candidate(character):
    crop = _binary_crop(character)
    if crop is None:
        return False
    h, w = crop.shape
    if h == 0 or not (0.90 <= w / h <= 1.10):
        return False
    n, _, _, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n - 1 != 2:
        return False
    foreground = cv2.resize(crop, (40, 40), interpolation=cv2.INTER_NEAREST) > 0
    horizontal = float(np.mean(foreground[17:23, :]))
    vertical = float(np.mean(foreground[:, 17:23]))
    corners = [foreground[:10, :10], foreground[:10, -10:],
               foreground[-10:, :10], foreground[-10:, -10:]]
    corner_density = float(np.mean([np.mean(c) for c in corners]))
    return (_asterisk(crop) and horizontal >= 0.60 and vertical >= 0.40
            and corner_density <= 0.05)


def predict_digit(character, model):
    processed = preprocess(character)
    if processed is None:
        raise ValueError('Empty character')
    probs = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])
    if confidence < 0.80:
        for dx in (-1, -2):
            other = shifted(processed, dx)
            p = model.predict(other.reshape(1, 28, 28, 1), verbose=0)[0]
            d = int(np.argmax(p))
            if float(p[d]) > confidence:
                digit, confidence = d, float(p[d])
    return str(digit)


def recognise(chars, split_flags, model, paren_flags=None, experimental=False):
    if paren_flags is None:
        paren_flags = [None] * len(chars)
    if len(chars) != len(split_flags) or len(chars) != len(paren_flags):
        raise ValueError('Flag alignment mismatch')
    symbols = []
    for char, from_split, paren in zip(chars, split_flags, paren_flags):
        if experimental and paren:
            symbols.append(paren)
            continue
        if experimental and not from_split and star_candidate(char):
            symbols.append('*')
            continue
        op = None if from_split else recognise_operator(char)
        symbols.append(op if op is not None else predict_digit(char, model))
    return ''.join(symbols)


def box_key(box):
    return tuple(int(v) for v in box)


def main():
    if not MODEL.is_file() or not IMAGES.is_dir():
        raise FileNotFoundError('Missing model or member_sources/member_2/testimages')
    model = load_model(str(MODEL))
    paths = sorted((p for p in IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    print(f'Found {len(paths)} images. V4.2 vs V4.3 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"V4.2":<23}{"V4.3":<23}Changes')
    print('-' * 105)
    rows = []
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        row = dict(testcase=num, expected=EXPECTED.get(num, ''), v42='', v43='',
                   changes='', masks=0, error='')
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot read image')
            chars, boxes = segment_digits(image, method='improved', group_division=True)
            chars, boxes = merge_division_parts(chars, boxes)
            base_chars, _, base_flags = split_connected_digits(
                chars, boxes, return_split_flags=True)
            row['v42'] = recognise(base_chars, base_flags, model)

            masked, masked_boxes, original_parens = isolate_parentheses(
                chars, boxes, return_flags=True)
            row['masks'] = sum(p is not None for p in original_parens)
            new_chars, new_boxes, new_flags = split_connected_digits(
                masked, masked_boxes, return_split_flags=True)
            # Map only unmodified parenthesis boxes. If a crop is split,
            # do not propagate the parenthesis flag to either digit piece.
            paren_by_box = {box_key(b): p for b, p in zip(masked_boxes, original_parens) if p}
            aligned_parens = [None if flag else paren_by_box.get(box_key(box))
                              for box, flag in zip(new_boxes, new_flags)]
            row['v43'] = recognise(new_chars, new_flags, model, aligned_parens, True)
            row['changes'] = ', '.join(
                f'{i}:{a}->{b}' for i, (a, b) in enumerate(zip(row['v42'], row['v43'])) if a != b
            ) if len(row['v42']) == len(row['v43']) else 'token positions changed'
            print(f'{num:<6}{(row["expected"] or "unknown"):<22}'
                  f'{row["v42"]:<23}{row["v43"]:<23}{row["changes"]}')
        except Exception as exc:
            row['error'] = str(exc)
            print(f'{num:<6}ERROR: {exc}')
        rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    labelled = [r for r in rows if r['expected'] and not r['error']]
    base = sum(r['v42'] == r['expected'] for r in labelled)
    new = sum(r['v43'] == r['expected'] for r in labelled)
    gains = [r['testcase'] for r in labelled if r['v42'] != r['expected'] and r['v43'] == r['expected']]
    regressions = [r['testcase'] for r in labelled if r['v42'] == r['expected'] and r['v43'] != r['expected']]
    print(f'\nV4.2: {base}/{len(labelled)} | V4.3: {new}/{len(labelled)}')
    print(f'Newly correct: {gains} | Regressions: {regressions}')
    print('Testcase10:', next(((r['v42'], r['v43']) for r in rows if r['testcase'] == 10), 'not found'))
    print('Errors:', [(r['testcase'], r['error']) for r in rows if r['error']])
    print('CSV:', OUTPUT)
    print('No working project files modified.')


if __name__ == '__main__':
    main()
