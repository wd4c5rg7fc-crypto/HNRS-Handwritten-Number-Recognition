"""Read-only V5.0 vs V5.1 narrow-opening-parenthesis experiment.

Run:
    python compare_narrow_paren_v51.py --new-dir test_images/new_tests

Only writes test_results/v51_narrow_paren_comparison.csv.
All recognition dependencies must be beside this file in the project root.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import compare_slash_v50 as v50

base = v50.base
OUT = Path(__file__).resolve().parent / 'test_results' / 'v51_narrow_paren_comparison.csv'


def narrow_open_paren_geometry(char):
    """Measure leftward curvature of a slim '(', without requiring a wide aspect ratio.

    Returns diagnostic measurements so the thresholds can be calibrated without
    hard-coding a testcase or blindly converting '1' after division.
    """
    mask = v50.v49.v48.v47.experiment.mask_of(char)
    if mask is None:
        return False, 'empty'
    h, w = mask.shape
    if h < 25 or w < 5 or not 0.09 <= w / h <= 0.85:
        return False, f'aspect={w/h:.2f} size={w}x{h}'
    centers = []
    for lo, hi in ((0.08, 0.28), (0.42, 0.58), (0.72, 0.92)):
        band = mask[int(lo*h):max(int(hi*h), int(lo*h)+1)]
        _, xs = np.nonzero(band)
        if len(xs) < 3:
            return False, f'sparse-band size={w}x{h}'
        centers.append(float(np.median(xs)))
    top, mid, bottom = centers
    # An opening parenthesis curves left at its midpoint; a slash tends to
    # move monotonically across the rows. Use pixels as well as relative bend.
    bend_px = (top + bottom) / 2 - mid
    bend_norm = bend_px / w
    balanced_ends = abs(top - bottom) / w
    # Reject mostly straight or slanted strokes and one-sided hooks.
    ok = (bend_px >= 2.0 and bend_norm >= 0.13
          and top - mid >= 1.5 and bottom - mid >= 1.0
          and balanced_ends <= 0.48)
    return ok, (f'size={w}x{h} aspect={w/h:.2f} '
                f'left-bend={bend_norm:.2f} bend-px={bend_px:.1f} '
                f'end-difference={balanced_ends:.2f} '
                f'centers={[round(v, 1) for v in centers]}')


def recognise_v51(chars, boxes, flags, model, parens):
    old, diagnostics = v50.recognise_v50(chars, boxes, flags, model, parens)
    diagnostics = list(diagnostics)
    _, _, removed, _ = v50.v49.v48.v47.experiment.revised.recognise_revised(
        chars, boxes, flags, model, parens)
    keep = [i for i in range(len(chars)) if i not in removed]
    if (v50.v49.v48.v47.experiment.v45.edge_fragment_candidate(
            0, chars, boxes, flags, parens) and 0 in keep):
        keep.remove(0)
    if len(old) != len(keep):
        return old, diagnostics + ['V5.1 skipped: alignment mismatch']
    symbols = list(old)
    for j in range(1, len(symbols)-1):
        i = keep[j]
        # Only after an already recognised division operator, before a digit.
        if (symbols[j] != '1' or symbols[j-1] != '/'
                or not symbols[j+1].isdigit() or flags[i] or parens[i]):
            continue
        accepted, why = narrow_open_paren_geometry(chars[i])
        diagnostics.append(f'narrow opening candidate {i}: {accepted} ({why})')
        if accepted:
            symbols[j] = '('
            diagnostics.append(f'REPLACED segment {i}: 1 -> (')
    return ''.join(symbols), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-dir', type=Path)
    args = parser.parse_args()
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        raise FileNotFoundError('Missing model or original testcase image directory')
    model = base.load_model(str(base.MODEL))
    originals = sorted((p for p in base.IMAGES.glob('testcase*.*')
                        if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                       key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    items = [('original', p, base.EXPECTED.get(int(''.join(c for c in p.stem if c.isdigit()) or 0), ''))
             for p in originals]
    if args.new_dir:
        if not args.new_dir.is_dir():
            parser.error(f'New image directory not found: {args.new_dir}')
        # User's files have doubled extensions, so match by their existing names.
        known = {'new_test_01.png.png': '18*(12-7)+36/9',
                 'new_test_02.png.png': '200-(48/6)*7',
                 'new_test_03.png.png': '(100-25)/(5+10)*3'}
        items.extend(('new', p, known.get(p.name, '')) for p in sorted(args.new_dir.iterdir())
                     if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    rows = []
    for group, path, expected in items:
        row = dict(group=group, image=path.name, expected=expected,
                   v50='', v51='', diagnostics='', error='')
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot read image')
            chars, boxes, flags, parens = v50.segment(image)
            row['v50'], _ = v50.recognise_v50(chars, boxes, flags, model, parens)
            row['v51'], diagnostics = recognise_v51(chars, boxes, flags, model, parens)
            row['diagnostics'] = '; '.join(diagnostics)
        except Exception as exc:
            row['error'] = repr(exc)
        rows.append(row)
        print(f'{group:8} {path.name:27} expected={expected or "?":22} '
              f'V5.0={row["v50"]:22} V5.1={row["v51"]:22} '
              f'{row["error"] or row["diagnostics"]}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for group in ('original', 'new'):
        subset = [r for r in rows if r['group'] == group and r['expected'] and not r['error']]
        before = sum(r['v50'] == r['expected'] for r in subset)
        after = sum(r['v51'] == r['expected'] for r in subset)
        regressions = [r['image'] for r in subset if r['v50'] == r['expected'] and r['v51'] != r['expected']]
        print(f'{group}: V5.0 {before}/{len(subset)} | V5.1 {after}/{len(subset)} | regressions={regressions}')
    print('Changed:', [r['image'] for r in rows if r['v50'] != r['v51'] and not r['error']])
    print('Errors:', [(r['image'], r['error']) for r in rows if r['error']])
    print('CSV:', OUT)
    print('Read-only: no GUI, model, or input images modified.')


if __name__ == '__main__':
    main()
