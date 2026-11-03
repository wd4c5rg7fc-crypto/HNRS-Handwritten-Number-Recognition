"""Read-only V5.2 vs experimental V5.3 (handwritten 4 vs +).

Run:
  python compare_four_plus_v53.py --new-dir test_images/new_tests --extra-image "path/to/144_expression.png"

The extra image must be the ORIGINAL handwritten image, not a GUI screenshot.
Writes only test_results/v53_four_plus_comparison.csv.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import compare_narrow_paren_v52 as v52

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'test_results' / 'v53_four_plus_comparison.csv'
base = v52.base


def four_vs_plus_geometry(char):
    """Test for a digit-4's upper diagonal, distinct from a centered plus stem.

    This is a hypothesis to validate, NOT a guaranteed classifier.
    """
    mask = v52.v51.v50.v49.v48.v47.experiment.mask_of(char)
    if mask is None:
        return False, 'empty'
    h, w = mask.shape
    if h < 12 or w < 9:
        return False, f'too small {w}x{h}'
    mask = (mask > 0).astype(np.uint8)
    # Identify the most horizontal row in the central portion (the crossbar).
    lo, hi = max(1, int(h*.28)), min(h-1, int(h*.77))
    bar_y = max(range(lo, hi), key=lambda y: int(mask[y].sum()))
    bar_ratio = float(mask[bar_y].sum()) / w
    if bar_ratio < .45 or bar_y < 4:
        return False, f'no substantial crossbar: {bar_ratio:.2f}'
    # A plus typically has a near-vertical upper arm in the center.
    # A handwritten 4 may have an oblique upper-left arm, plus a right stem.
    upper = mask[:bar_y]
    points = [(y, float(np.median(np.flatnonzero(upper[y]))))
              for y in range(upper.shape[0]) if np.count_nonzero(upper[y])]
    if len(points) < 4:
        return False, f'upper stroke sparse: rows={len(points)}'
    top = np.median([x for y,x in points if y < max(2, bar_y*.40)]) if any(y < max(2,bar_y*.40) for y,x in points) else points[0][1]
    lower = np.median([x for y,x in points if y >= bar_y*.60]) if any(y >= bar_y*.60 for y,x in points) else points[-1][1]
    displacement = abs(float(lower-top))/w
    # A genuine 4 often has at least two separated ink groups near the upper
    # crossbar; plus typically only has a single central vertical arm.
    dual_rows = 0
    for row in upper[max(0,int(bar_y*.35)):]:
        xs = np.flatnonzero(row)
        if len(xs) >= 2 and np.max(np.diff(xs)) >= max(3, int(w*.22)):
            dual_rows += 1
    dual_fraction = dual_rows / max(1, bar_y-int(bar_y*.35))
    accepted = (displacement >= .13 and dual_fraction >= .13 and bar_ratio >= .45)
    return accepted, (f'size={w}x{h} bar-y={bar_y} bar={bar_ratio:.2f} '
                      f'upper-displacement={displacement:.2f} '
                      f'dual-upper={dual_fraction:.2f}')


def recognise_v53(chars, boxes, flags, model, parens):
    expression, diag = v52.recognise_v52(chars, boxes, flags, model, parens)
    diag = list(diag)
    _, _, removed, _ = v52.v51.v50.v49.v48.v47.experiment.revised.recognise_revised(
        chars, boxes, flags, model, parens)
    keep = [i for i in range(len(chars)) if i not in removed]
    if (v52.v51.v50.v49.v48.v47.experiment.v45.edge_fragment_candidate(
            0, chars, boxes, flags, parens) and 0 in keep):
        keep.remove(0)
    if len(keep) != len(expression):
        return expression, diag + ['V5.3 skipped: alignment mismatch']
    symbols = list(expression)
    for j in range(1, len(symbols)-1):
        i = keep[j]
        if symbols[j] != '+' or flags[i] or parens[i]:
            continue
        # Syntax gate: '+' immediately after '*' or '/' cannot be a binary plus.
        # Geometry must independently support a '4'; never correct from syntax alone.
        if symbols[j-1] not in '*/' or symbols[j+1] not in '0123456789)':
            continue
        yes, why = four_vs_plus_geometry(chars[i])
        diag.append(f'V5.3 4-vs-plus candidate {i}: {yes} ({why})')
        if yes:
            symbols[j] = '4'
            diag.append(f'V5.3 REPLACED segment {i}: + -> 4')
    return ''.join(symbols), diag


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--new-dir', type=Path)
    ap.add_argument('--extra-image', type=Path,
                    help='Original source image of 144/(6*4)+17, not screenshot or crop')
    args = ap.parse_args()
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        ap.error('Model or original image directory missing')
    model = base.load_model(str(base.MODEL))
    originals = sorted((p for p in base.IMAGES.glob('testcase*.*')
                        if p.suffix.lower() in ('.png','.jpg','.jpeg')),
                       key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    items = [('original', p, base.EXPECTED.get(
        int(''.join(c for c in p.stem if c.isdigit()) or 0), '')) for p in originals]
    if args.new_dir:
        if not args.new_dir.is_dir():
            ap.error(f'Not found: {args.new_dir}')
        known = {'new_test_01':'18*(12-7)+36/9',
                 'new_test_02':'200-(48/6)*7',
                 'new_test_03':'(100-25)/(5+10)*3'}
        for p in sorted(args.new_dir.iterdir()):
            if p.suffix.lower() not in ('.png','.jpg','.jpeg'):
                continue
            name = p.name
            while Path(name).suffix.lower() in ('.png','.jpg','.jpeg'):
                name = Path(name).stem
            items.append(('new', p, known.get(name, '')))
    if args.extra_image:
        if not args.extra_image.is_file():
            ap.error(f'Extra image not found: {args.extra_image}')
        items.append(('extra', args.extra_image, '144/(6*4)+17'))
    rows = []
    for group, p, expected in items:
        row = dict(group=group, image=str(p), expected=expected,
                   v52='', v53='', diagnostics='', error='')
        try:
            im = cv2.imread(str(p))
            if im is None:
                raise ValueError('Cannot read image')
            chars, boxes, flags, parens = v52.v51.v50.segment(im)
            row['v52'], _ = v52.recognise_v52(chars, boxes, flags, model, parens)
            row['v53'], details = recognise_v53(chars, boxes, flags, model, parens)
            row['diagnostics'] = '; '.join(details)
        except Exception as exc:
            row['error'] = repr(exc)
        rows.append(row)
        print(f'{group:8} {p.name:28} expected={expected or "?":22} '
              f'V5.2={row["v52"]:22} V5.3={row["v53"]:22} '
              f'{row["error"] or row["diagnostics"]}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for group in ('original','new','extra'):
        subset = [r for r in rows if r['group']==group and r['expected'] and not r['error']]
        if not subset:
            continue
        before = sum(r['v52']==r['expected'] for r in subset)
        after = sum(r['v53']==r['expected'] for r in subset)
        regressions = [r['image'] for r in subset
                       if r['v52']==r['expected'] and r['v53']!=r['expected']]
        print(f'{group}: V5.2 {before}/{len(subset)} | V5.3 {after}/{len(subset)} '
              f'| regressions={regressions}')
    print('Changed:', [r['image'] for r in rows if r['v52']!=r['v53'] and not r['error']])
    print('Errors:', [(r['image'],r['error']) for r in rows if r['error']])
    print('CSV:', OUT)
    print('Read-only: no GUI, model, or input images modified.')


if __name__ == '__main__':
    main()
