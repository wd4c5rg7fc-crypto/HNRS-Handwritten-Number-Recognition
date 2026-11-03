"""Read-only diagnostics for the leading-fragment filter in revised V4.5.

Run: python diagnose_testcase14_fragment.py
Requires existing compare_testcase14_v45_revised.py and its dependencies.
Prints every predicate for every image; does not alter recognizers or GUI.
"""
import csv
from pathlib import Path
import cv2
import compare_testcase14_v45_revised as revised

base = revised.base
OUTPUT = Path(__file__).resolve().parent / 'test_results' / 'leading_fragment_diagnostics.csv'


def check_first_pair(chars, boxes, flags, parens):
    if len(chars) < 2:
        return {'reason': 'fewer than 2 characters'}
    first = base._binary_crop(chars[0])
    second = base._binary_crop(chars[1])
    if first is None or second is None:
        return {'reason': 'empty binary crop'}
    h0, w0 = first.shape
    h1, w1 = second.shape
    a0, a1 = cv2.countNonZero(first), cv2.countNonZero(second)
    x0, y0, bw0, bh0 = map(int, boxes[0])
    x1, y1, bw1, bh1 = map(int, boxes[1])
    gap = x1 - (x0 + bw0)
    checks = {
        'not_split': not bool(flags[0]),
        'not_paren': not bool(parens[0]),
        'second_height_ge_45': h1 >= 45,
        'second_slender': w1 / max(h1, 1) <= .30,
        'first_height_ratio': h0 / max(h1, 1) <= .45,
        'first_area_ratio': a0 / max(a1, 1) <= .30,
        'first_to_left': x0 < x1,
        'nonnegative_gap': gap >= 0,
        'gap_within_limit': gap <= max(40, .85 * h1),
        'vertical_overlap_top': y0 < y1 + bh1,
        'vertical_overlap_bottom': y0 + bh0 > y1,
    }
    result = {
        'box0': str(tuple(map(int, boxes[0]))),
        'box1': str(tuple(map(int, boxes[1]))),
        'crop0': f'{h0}x{w0}', 'crop1': f'{h1}x{w1}',
        'area0': a0, 'area1': a1,
        'height_ratio': round(h0 / max(h1, 1), 4),
        'area_ratio': round(a0 / max(a1, 1), 4),
        'slender_ratio': round(w1 / max(h1, 1), 4),
        'gap': gap,
        'checks': checks,
        'original_candidate': revised.leading_fragment_candidate(0, chars, boxes, flags, parens),
    }
    return result


def main():
    paths = sorted(
        (p for p in base.IMAGES.glob('testcase*.*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
        key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0),
    )
    rows = []
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        image = cv2.imread(str(path))
        if image is None:
            print(f'{num:2}: cannot read image')
            continue
        chars, boxes = base.segment_digits(image, method='improved', group_division=True)
        chars, boxes = base.merge_division_parts(chars, boxes)
        masked, masked_boxes, parens = base.isolate_parentheses(chars, boxes, return_flags=True)
        chars, boxes, flags = base.split_connected_digits(masked, masked_boxes, return_split_flags=True)
        by_box = {base.box_key(b): p for b, p in zip(masked_boxes, parens) if p}
        aligned = [None if flag else by_box.get(base.box_key(box)) for box, flag in zip(boxes, flags)]
        info = check_first_pair(chars, boxes, flags, aligned)
        if 'checks' not in info:
            print(f'{num:2}: {info["reason"]}')
            continue
        failed = [key for key, passed in info['checks'].items() if not passed]
        print(f'{num:2}: candidate={info["original_candidate"]!s:<5} '
              f'failed={failed} gap={info["gap"]} '
              f'height_ratio={info["height_ratio"]} area_ratio={info["area_ratio"]}')
        if num == 14:
            print('\n=== TESTCASE14 FULL DIAGNOSTIC ===')
            for key, value in info.items():
                if key == 'checks':
                    for condition, passed in value.items():
                        print(f'  {"PASS" if passed else "FAIL"}  {condition}')
                else:
                    print(f'  {key}: {value}')
            print('=== END TESTCASE14 ===\n')
        row = {'testcase': num, **{k: v for k, v in info.items() if k != 'checks'},
               'failed_checks': ';'.join(failed)}
        row.update({f'check_{k}': v for k, v in info['checks'].items()})
        rows.append(row)
    if rows:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('\nCSV:', OUTPUT)
    print('Read-only diagnostic. No GUI or model files changed.')


if __name__ == '__main__':
    main()
