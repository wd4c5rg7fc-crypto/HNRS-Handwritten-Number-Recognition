"""Read-only V4.4 versus V4.5 edge-fragment experiment.

Place alongside compare_testcase14_v45_revised.py and its dependencies.
Run: python compare_testcase14_v45_edge.py
No GUI, segmentation, recognizer, or model files are modified.
"""
import csv
from pathlib import Path
import cv2
import compare_testcase14_v45_revised as revised

base = revised.base
v44 = revised.v44
OUTPUT = Path(__file__).resolve().parent / 'test_results' / 'testcase14_v45_edge_comparison.csv'


def edge_fragment_candidate(index, chars, boxes, flags, parens):
    """Flag substantial but small marks far left and vertically outside text.

    Uses image geometry, never testcase IDs or expected expressions.
    Deliberately avoids ultra-tiny specks pending review of unlabelled images.
    """
    if index != 0 or len(chars) < 3 or flags[0] or parens[0]:
        return False
    first = base._binary_crop(chars[0])
    if first is None:
        return False
    x0, y0, w0, h0 = map(int, boxes[0])
    area0 = cv2.countNonZero(first)
    other = [tuple(map(int, b)) for b in boxes[1:]]
    if not other:
        return False
    main_left = min(b[0] for b in other)
    main_top = min(b[1] for b in other)
    main_bottom = max(b[1] + b[3] for b in other)
    heights = sorted(b[3] for b in other)
    typical_height = heights[len(heights) // 2]
    gap = main_left - (x0 + w0)
    vertical_separation = max(main_top - (y0 + h0), y0 - main_bottom, 0)
    return (
        80 <= area0 <= 0.30 * max(cv2.countNonZero(base._binary_crop(c))
                                    for c in chars[1:] if base._binary_crop(c) is not None)
        and 15 <= h0 <= 0.55 * typical_height
        and w0 <= 0.55 * typical_height
        and x0 <= 0.05 * max(b[0] + b[2] for b in boxes)
        and gap >= 2.0 * typical_height
        and vertical_separation >= 0.75 * typical_height
    )


def main():
    model = base.load_model(str(base.MODEL))
    paths = sorted((p for p in base.IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    rows = []
    print(f'Found {len(paths)} images. V4.4 vs edge-filter V4.5 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"V4.4":<23}{"V4.5 edge":<23}Removed / stars')
    print('-' * 95)
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        row = dict(testcase=num, expected=base.EXPECTED.get(num, ''),
                   v44='', v45_edge='', removed='', stars='', error='')
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
            # Retain the previously tested safe CNN-4 radial-star rule.
            text, stars, removed, _ = revised.recognise_revised(chars, boxes, flags, model, aligned)
            if edge_fragment_candidate(0, chars, boxes, flags, aligned):
                if 0 not in removed:
                    text = text[1:]
                    removed.append(0)
            row['v45_edge'] = text
            row['removed'] = ','.join(map(str, removed))
            row['stars'] = ','.join(map(str, stars))
            print(f'{num:<6}{(row["expected"] or "unknown"):<22}{row["v44"]:<23}'
                  f'{text:<23}{removed} / {stars}')
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
    new = sum(r['v45_edge'] == r['expected'] for r in labelled)
    gains = [r['testcase'] for r in labelled if r['v44'] != r['expected'] and r['v45_edge'] == r['expected']]
    regressions = [r['testcase'] for r in labelled if r['v44'] == r['expected'] and r['v45_edge'] != r['expected']]
    print(f'\nV4.4: {old}/{len(labelled)} | V4.5 edge: {new}/{len(labelled)}')
    print('Newly correct:', gains, '| Regressions:', regressions)
    print('Errors:', [(r['testcase'], r['error']) for r in rows if r['error']])
    print('CSV:', OUTPUT)
    print('No working project files modified.')


if __name__ == '__main__':
    main()
