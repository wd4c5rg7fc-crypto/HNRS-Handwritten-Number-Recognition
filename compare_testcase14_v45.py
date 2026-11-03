"""Read-only V4.4 vs experimental V4.5 comparison for testcase14.

Run from project root: python compare_testcase14_v45.py
Requires compare_testcase20_v44.py and its existing dependencies.
Does not change app.py, recognizers, segmentation, or the CNN model.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
import compare_testcase20_v44 as v44

base = v44.baseline
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'test_results' / 'testcase14_v45_comparison.csv'


def radial_star_candidate(character):
    """Experimental single-piece radial asterisk detector, not a trained classifier."""
    crop = base._binary_crop(character)
    if crop is None:
        return False
    h, w = crop.shape
    if min(h, w) < 35 or not (0.80 <= w / h <= 1.12):
        return False
    n, _, _, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n - 1 != 1:
        return False
    mask = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.where(mask)
    if len(xs) < 150:
        return False
    cx, cy = float(np.median(xs)), float(np.median(ys))
    yy, xx = np.indices(mask.shape)
    radius = np.hypot(xx - cx, yy - cy)
    angles = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    hub_density = float(np.mean(mask[radius <= 7]))
    occupied = sum(np.count_nonzero(mask & (radius >= 15) & (radius <= 29)
                                   & (angles >= i * np.pi / 6)
                                   & (angles < (i + 1) * np.pi / 6)) >= 15
                   for i in range(12))
    return hub_density >= 0.70 and occupied >= 10


def artifact_candidate(index, chars, boxes, flags, parens):
    """Only remove an isolated, tiny leading fragment next to a full-height stroke.

    No testcase numbers or expected strings are used in this decision.
    """
    if index != 0 or len(chars) < 2 or flags[index] or parens[index]:
        return False
    first = base._binary_crop(chars[0])
    second = base._binary_crop(chars[1])
    if first is None or second is None:
        return False
    h0, w0 = first.shape
    h1, w1 = second.shape
    a0 = cv2.countNonZero(first)
    a1 = cv2.countNonZero(second)
    x0, y0, bw0, bh0 = map(int, boxes[0])
    x1, y1, bw1, bh1 = map(int, boxes[1])
    return (h0 <= 0.40 * h1 and a0 <= 0.28 * a1
            and w0 <= 2.0 * w1 and x0 < x1
            and x1 - (x0 + bw0) <= max(20, 0.6 * h1)
            and abs((y0 + bh0 / 2) - (y1 + bh1 / 2)) <= h1)


def recognise_v45(chars, boxes, split_flags, model, paren_flags):
    if not (len(chars) == len(boxes) == len(split_flags) == len(paren_flags)):
        raise ValueError('Input lengths differ')
    symbols, star_hits, removed = [], [], []
    for i, (char, from_split, paren) in enumerate(zip(chars, split_flags, paren_flags)):
        if artifact_candidate(i, chars, boxes, split_flags, paren_flags):
            removed.append(i)
            continue
        if paren:
            symbols.append(paren)
        elif not from_split and v44.division_candidate(char):
            symbols.append('/')
        elif not from_split and base.star_candidate(char):
            symbols.append('*')
        elif not from_split and radial_star_candidate(char):
            symbols.append('*')
            star_hits.append(i)
        else:
            op = None if from_split else base.recognise_operator(char)
            symbols.append(op if op is not None else base.predict_digit(char, model))
    return ''.join(symbols), star_hits, removed


def main():
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        raise FileNotFoundError('Missing model or member_2/testimages')
    model = base.load_model(str(base.MODEL))
    paths = sorted((p for p in base.IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    print(f'Found {len(paths)} images. V4.4 vs experimental V4.5 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"V4.4":<23}{"V4.5":<23}Star / removed')
    print('-' * 105)
    rows = []
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        row = dict(testcase=num, expected=base.EXPECTED.get(num, ''), v44='', v45='',
                   star_matches='', removed_fragments='', error='')
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot read image')
            chars, boxes = base.segment_digits(image, method='improved', group_division=True)
            chars, boxes = base.merge_division_parts(chars, boxes)
            masked, masked_boxes, parens = base.isolate_parentheses(chars, boxes, return_flags=True)
            chars, boxes, flags = base.split_connected_digits(masked, masked_boxes,
                                                              return_split_flags=True)
            by_box = {base.box_key(b): p for b, p in zip(masked_boxes, parens) if p}
            aligned = [None if flag else by_box.get(base.box_key(box))
                       for box, flag in zip(boxes, flags)]
            row['v44'], _ = v44.recognise_v44(chars, flags, model, aligned)
            row['v45'], hits, removed = recognise_v45(chars, boxes, flags, model, aligned)
            row['star_matches'] = ','.join(map(str, hits))
            row['removed_fragments'] = ','.join(map(str, removed))
            print(f'{num:<6}{(row["expected"] or "unknown"):<22}{row["v44"]:<23}'
                  f'{row["v45"]:<23}{hits} / {removed}')
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
    old = sum(r['v44'] == r['expected'] for r in labelled)
    new = sum(r['v45'] == r['expected'] for r in labelled)
    gains = [r['testcase'] for r in labelled if r['v44'] != r['expected'] and r['v45'] == r['expected']]
    regressions = [r['testcase'] for r in labelled if r['v44'] == r['expected'] and r['v45'] != r['expected']]
    print(f'\nV4.4: {old}/{len(labelled)} | V4.5: {new}/{len(labelled)}')
    print('Newly correct:', gains, '| Regressions:', regressions)
    print('Errors:', [(r['testcase'], r['error']) for r in rows if r['error']])
    print('CSV:', OUTPUT)
    print('No working project files modified.')


if __name__ == '__main__':
    main()
