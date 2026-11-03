"""Read-only V4.6 vs exploratory V4.7 on all testcase images.

Place beside compare_testcase13_v46_revised.py and dependencies.
Run: python compare_testcase15_v47.py
Does not edit app.py, recognizer modules, models, or source images.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
import compare_testcase13_v46 as experiment
from compare_testcase13_v46_revised import open_paren_geometry_revised

# Restore the validated V4.6 rule in memory only.
experiment.open_paren_geometry = open_paren_geometry_revised
base = experiment.base
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'test_results' / 'testcase15_v47_comparison.csv'


def star_geometry(char):
    """Hypothesis: broad, multi-armed asterisk rather than a narrow handwritten 5."""
    m = experiment.mask_of(char)
    if m is None:
        return False, 'empty'
    h, w = m.shape
    ratio = w / h
    # Testcase15's star is substantially wider than tall, unlike typical 5s.
    if min(h, w) < 30 or not 1.15 <= ratio <= 2.1:
        return False, f'aspect={ratio:.2f}'
    # At least three distinct directions of ink must reach the outer perimeter.
    norm = cv2.resize(m, (80, 80), interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.where(norm)
    if len(xs) < 100:
        return False, 'sparse'
    cx, cy = float(np.median(xs)), float(np.median(ys))
    yy, xx = np.indices(norm.shape)
    r = np.hypot(xx - cx, yy - cy)
    theta = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    sectors = [int(np.count_nonzero(norm & (r >= 19) & (r <= 46) &
                                      (theta >= i * np.pi / 4) &
                                      (theta < (i + 1) * np.pi / 4)))
               for i in range(8)]
    arms = sum(count >= 18 for count in sectors)
    ok = arms >= 5
    return ok, f'aspect={ratio:.2f} arms={arms}/8 sectors={sectors}'


def closing_paren_geometry(char):
    """Hypothesis: ')' bulges right at mid-height, unlike a straight-stroke 7."""
    m = experiment.mask_of(char)
    if m is None:
        return False, 'empty'
    h, w = m.shape
    if h < 45 or w < 20 or not .25 <= w / h <= .90:
        return False, f'aspect={w/h:.2f}'
    centers = []
    for lo, hi in ((.10, .25), (.42, .58), (.75, .90)):
        band = m[int(h * lo):max(int(h * hi), int(h * lo) + 1)]
        _, xs = np.nonzero(band)
        if len(xs) < 10:
            return False, 'sparse-band'
        centers.append(float(np.median(xs)) / w)
    top, mid, bottom = centers
    bend = mid - (top + bottom) / 2
    ok = bend >= .12 and mid - top >= .15 and abs(top - bottom) <= .45
    return ok, f'bend={bend:.2f} centers={[round(x,2) for x in centers]}'


def recognise_v47(chars, boxes, flags, model, parens):
    """Apply conservative, contextual changes to the validated V4.6 output."""
    old, v46_diag = experiment.recognise_v46(chars, boxes, flags, model, parens)
    _, _, removed, _ = experiment.revised.recognise_revised(chars, boxes, flags, model, parens)
    keep = [i for i in range(len(chars)) if i not in removed]
    if experiment.v45.edge_fragment_candidate(0, chars, boxes, flags, parens) and 0 in keep:
        keep.remove(0)
    diagnostics = list(v46_diag)
    if len(old) != len(keep):
        return old, diagnostics + ['skipped: segment alignment mismatch']
    result = list(old)

    # Candidate '*' must be between an existing digit and an opening '('.
    for j in range(1, len(result) - 1):
        i = keep[j]
        if result[j] != '5' or not result[j - 1].isdigit() or result[j + 1] != '(':
            continue
        if flags[i] or parens[i]:
            continue
        yes, why = star_geometry(chars[i])
        diagnostics.append(f'star at {i}: {yes} ({why})')
        if yes:
            result[j] = '*'
            diagnostics.append(f'REPLACED {i}: 5 -> *')

    # Candidate ')' must be the final character and close an unmatched '('.
    if result and result[-1] == '7' and result.count('(') > result.count(')'):
        i = keep[-1]
        if not flags[i] and not parens[i]:
            yes, why = closing_paren_geometry(chars[i])
            diagnostics.append(f'closing paren at {i}: {yes} ({why})')
            if yes:
                result[-1] = ')'
                diagnostics.append(f'REPLACED {i}: 7 -> )')
    return ''.join(result), diagnostics


def main():
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        raise FileNotFoundError('Missing CNN model or test image directory')
    model = base.load_model(str(base.MODEL))
    paths = sorted((p for p in base.IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    rows = []
    print(f'Found {len(paths)} images. V4.6 vs experimental V4.7 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"V4.6":<23}{"V4.7":<23}Diagnostics')
    print('-' * 120)
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        row = dict(testcase=num, expected=base.EXPECTED.get(num, ''),
                   v46='', v47='', diagnostics='', error='')
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
            row['v46'], _ = experiment.recognise_v46(chars, boxes, flags, model, aligned)
            row['v47'], diag = recognise_v47(chars, boxes, flags, model, aligned)
            row['diagnostics'] = '; '.join(diag)
            print(f'{num:<6}{(row["expected"] or "unknown"):<22}'
                  f'{row["v46"]:<23}{row["v47"]:<23}{row["diagnostics"]}')
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
    old = sum(r['v46'] == r['expected'] for r in labelled)
    new = sum(r['v47'] == r['expected'] for r in labelled)
    gains = [r['testcase'] for r in labelled
             if r['v46'] != r['expected'] and r['v47'] == r['expected']]
    regressions = [r['testcase'] for r in labelled
                   if r['v46'] == r['expected'] and r['v47'] != r['expected']]
    changed = [r['testcase'] for r in rows if r['v46'] != r['v47'] and not r['error']]
    print(f'\nV4.6: {old}/{len(labelled)} | V4.7: {new}/{len(labelled)}')
    print('Newly correct:', gains, '| Regressions:', regressions)
    print('Changed images (including unlabelled):', changed)
    print('Errors:', [(r['testcase'], r['error']) for r in rows if r['error']])
    print('CSV:', OUTPUT)
    print('No working project files modified.')


if __name__ == '__main__':
    main()
