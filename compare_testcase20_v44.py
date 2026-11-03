"""Read-only V4.3 vs V4.4 experiment for testcase20.

Place beside compare_testcase10_v43.py in the project root.
Run: python compare_testcase20_v44.py
Does not modify GUI, operator_recognition.py, or trained model.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
import compare_testcase10_v43 as baseline

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'test_results' / 'testcase20_v44_comparison.csv'


def division_candidate(character):
    """Detect three substantial pieces (dot/bar/dot), ignoring tiny noise.

    The original V4.3 rule counts every foreground component, including
    tiny specks. This experiment filters components by relative area first.
    """
    crop = baseline._binary_crop(character)
    if crop is None:
        return False
    h, w = crop.shape
    if h < 24 or w < 24:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n < 4:
        return False  # Existing V4.3 handles ordinary three-part division.
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = int(areas.sum())
    parts = []
    for s in stats[1:]:
        x, y, pw, ph, area = map(int, s)
        if area >= max(8, 0.08 * total):
            parts.append((x, y, pw, ph, area))
    if len(parts) != 3:
        return False
    parts.sort(key=lambda p: p[1])
    top, middle, bottom = parts
    tx, ty, tw, th, ta = top
    mx, my, mw, mh, ma = middle
    bx, by, bw, bh, ba = bottom
    return (
        mw >= 2.3 * mh
        and mw >= 0.65 * w
        and ta < 0.75 * ma
        and ba < 0.75 * ma
        and tw < 0.55 * mw
        and bw < 0.55 * mw
        and ty + th <= my + 0.06 * h
        and my + mh <= by + 0.06 * h
        and ty < my < by
    )


def recognise_v44(chars, split_flags, model, paren_flags):
    if not (len(chars) == len(split_flags) == len(paren_flags)):
        raise ValueError('Character and flag counts do not match')
    symbols = []
    matches = []
    for i, (char, from_split, paren) in enumerate(zip(chars, split_flags, paren_flags)):
        if paren:
            symbols.append(paren)
        elif not from_split and division_candidate(char):
            symbols.append('/')
            matches.append(i)
        elif not from_split and baseline.star_candidate(char):
            symbols.append('*')
        else:
            op = None if from_split else baseline.recognise_operator(char)
            symbols.append(op if op is not None else baseline.predict_digit(char, model))
    return ''.join(symbols), matches


def main():
    if not baseline.MODEL.is_file() or not baseline.IMAGES.is_dir():
        raise FileNotFoundError('Missing CNN model or member_2/testimages directory')
    model = baseline.load_model(str(baseline.MODEL))
    paths = sorted((p for p in baseline.IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    print(f'Found {len(paths)} images. V4.3 vs experimental V4.4 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"V4.3":<23}{"V4.4":<23}Division matches')
    print('-' * 100)
    rows = []
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        row = dict(testcase=num, expected=baseline.EXPECTED.get(num, ''),
                   v43='', v44='', division_matches='', error='')
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot read image')
            chars, boxes = baseline.segment_digits(image, method='improved', group_division=True)
            chars, boxes = baseline.merge_division_parts(chars, boxes)
            masked, masked_boxes, parens = baseline.isolate_parentheses(chars, boxes, return_flags=True)
            new_chars, new_boxes, split_flags = baseline.split_connected_digits(
                masked, masked_boxes, return_split_flags=True)
            paren_by_box = {baseline.box_key(b): p for b, p in zip(masked_boxes, parens) if p}
            aligned = [None if flag else paren_by_box.get(baseline.box_key(box))
                       for box, flag in zip(new_boxes, split_flags)]
            row['v43'] = baseline.recognise(new_chars, split_flags, model, aligned, True)
            row['v44'], matches = recognise_v44(new_chars, split_flags, model, aligned)
            row['division_matches'] = ','.join(map(str, matches))
            print(f'{num:<6}{(row["expected"] or "unknown"):<22}'
                  f'{row["v43"]:<23}{row["v44"]:<23}{row["division_matches"]}')
        except Exception as exc:
            row['error'] = repr(exc)
            print(f'{num:<6}ERROR: {exc}')
        rows.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labelled = [r for r in rows if r['expected'] and not r['error']]
    old = sum(r['v43'] == r['expected'] for r in labelled)
    new = sum(r['v44'] == r['expected'] for r in labelled)
    gains = [r['testcase'] for r in labelled if r['v43'] != r['expected'] and r['v44'] == r['expected']]
    regressions = [r['testcase'] for r in labelled if r['v43'] == r['expected'] and r['v44'] != r['expected']]
    print(f'\nV4.3: {old}/{len(labelled)} | V4.4: {new}/{len(labelled)}')
    print('Newly correct:', gains, '| Regressions:', regressions)
    print('Errors:', [(r['testcase'], r['error']) for r in rows if r['error']])
    print('CSV:', OUTPUT)
    print('No working project files modified.')


if __name__ == '__main__':
    main()
