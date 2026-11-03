"""Read-only revised V4.5 experiment, preserving V4.4 as baseline.

Place alongside compare_testcase14_v45.py and compare_testcase20_v44.py.
Run: python compare_testcase14_v45_revised.py
Does not change the GUI, segmentation, model or recognizer files.
"""
import csv
from pathlib import Path
import cv2
import compare_testcase14_v45 as previous

v44 = previous.v44
base = previous.base
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'test_results' / 'testcase14_v45_revised_comparison.csv'


def leading_fragment_candidate(index, chars, boxes, flags, parens):
    """Conservative, geometry-only candidate for a tiny mark before a thin '1'.

    Unlike the first V4.5 experiment, require a very slender second glyph.
    Do not rely on the testcase number or expected expression.
    """
    if index != 0 or len(chars) < 2 or flags[0] or parens[0]:
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
    return (
        h1 >= 45
        and w1 / max(h1, 1) <= 0.30
        and h0 / h1 <= 0.45
        and a0 / max(a1, 1) <= 0.30
        and x0 < x1
        and 0 <= x1 - (x0 + bw0) <= max(40, 0.85 * h1)
        and y0 < y1 + bh1
        and y0 + bh0 > y1
    )


def recognise_revised(chars, boxes, flags, model, parens):
    if not (len(chars) == len(boxes) == len(flags) == len(parens)):
        raise ValueError('Character, box, and flag counts differ')
    symbols, stars, removed, diagnostics = [], [], [], []
    for i, (char, from_split, paren) in enumerate(zip(chars, flags, parens)):
        if leading_fragment_candidate(i, chars, boxes, flags, parens):
            removed.append(i)
            continue
        if paren:
            symbols.append(paren)
        elif not from_split and v44.division_candidate(char):
            symbols.append('/')
        elif not from_split and base.star_candidate(char):
            symbols.append('*')
        else:
            op = None if from_split else base.recognise_operator(char)
            if op is not None:
                symbols.append(op)
            else:
                digit = base.predict_digit(char, model)
                # The old radial rule misidentified 3 and 8. Only consider
                # reinterpreting a CNN-predicted 4, as in testcase14.
                if (not from_split and digit == '4'
                        and previous.radial_star_candidate(char)):
                    symbols.append('*')
                    stars.append(i)
                else:
                    symbols.append(digit)
                if not from_split and previous.radial_star_candidate(char):
                    diagnostics.append(f'{i}:cnn={digit}')
    return ''.join(symbols), stars, removed, diagnostics


def main():
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        raise FileNotFoundError('Missing model or member_2/testimages directory')
    model = base.load_model(str(base.MODEL))
    paths = sorted((p for p in base.IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    print(f'Found {len(paths)} images. V4.4 vs revised V4.5 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"V4.4":<23}{"Revised V4.5":<23}Star / removed / radial diagnostics')
    print('-' * 115)
    rows = []
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        row = dict(testcase=num, expected=base.EXPECTED.get(num, ''),
                   v44='', revised_v45='', star_matches='', removed_fragments='',
                   radial_diagnostics='', error='')
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
            row['revised_v45'], stars, removed, diagnostics = recognise_revised(
                chars, boxes, flags, model, aligned)
            row['star_matches'] = ','.join(map(str, stars))
            row['removed_fragments'] = ','.join(map(str, removed))
            row['radial_diagnostics'] = ';'.join(diagnostics)
            print(f'{num:<6}{(row["expected"] or "unknown"):<22}'
                  f'{row["v44"]:<23}{row["revised_v45"]:<23}'
                  f'{stars} / {removed} / {diagnostics}')
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
    new = sum(r['revised_v45'] == r['expected'] for r in labelled)
    gains = [r['testcase'] for r in labelled if r['v44'] != r['expected']
             and r['revised_v45'] == r['expected']]
    regressions = [r['testcase'] for r in labelled if r['v44'] == r['expected']
                   and r['revised_v45'] != r['expected']]
    print(f'\nV4.4: {old}/{len(labelled)} | Revised V4.5: {new}/{len(labelled)}')
    print('Newly correct:', gains, '| Regressions:', regressions)
    print('Errors:', [(r['testcase'], r['error']) for r in rows if r['error']])
    print('CSV:', OUTPUT)
    print('No working project files modified.')


if __name__ == '__main__':
    main()
