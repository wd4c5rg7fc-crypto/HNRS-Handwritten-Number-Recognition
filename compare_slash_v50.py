"""Read-only V4.9b vs experimental V5.0 recognition.

Run from the project root:
    python compare_slash_v50.py
    python compare_slash_v50.py --new-dir test_images/new_tests

No GUI, models, source images, or recognition modules are modified.
Only writes test_results/v50_slash_comparison.csv.
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
import compare_open_zero_v49b as v49

base = v49.base
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'test_results' / 'v50_slash_comparison.csv'


def slash_geometry(char):
    """Compare top/bottom stroke centers, requiring a steep /-shaped diagonal."""
    m = v49.v48.v47.experiment.mask_of(char)
    if m is None:
        return False, 'empty'
    h, w = m.shape
    if h < 25 or w < 7:
        return False, f'too-small={w}x{h}'
    centers = []
    for lo, hi in ((.08, .30), (.70, .92)):
        band = m[int(lo*h):max(int(hi*h), int(lo*h)+1)]
        ys, xs = np.nonzero(band)
        if len(xs) < 5:
            return False, 'sparse-band'
        centers.append(float(np.median(xs)))
    top, bottom = centers
    dx = top - bottom  # positive for /, negative for backslash
    slope = dx / h
    aspect = w / h
    # Conservative initial thresholds. Calibrate on the full test set.
    ok = (.22 <= aspect <= 1.35 and slope >= .24 and dx >= .25*w)
    return ok, f'width={w} height={h} aspect={aspect:.2f} top-bottom-dx={dx:.1f} slope={slope:.2f}'


def recognise_v50(chars, boxes, flags, model, parens):
    old, diag = v49.recognise_v49(chars, boxes, flags, model, parens)
    diag = list(diag)
    _, _, removed, _ = v49.v48.v47.experiment.revised.recognise_revised(
        chars, boxes, flags, model, parens)
    keep = [i for i in range(len(chars)) if i not in removed]
    if (v49.v48.v47.experiment.v45.edge_fragment_candidate(
            0, chars, boxes, flags, parens) and 0 in keep):
        keep.remove(0)
    if len(old) != len(keep):
        return old, diag + ['V5.0 skipped: alignment mismatch']
    symbols = list(old)
    for j in range(1, len(symbols)-1):
        i = keep[j]
        if symbols[j] != '1' or flags[i] or parens[i]:
            continue
        prev, nxt = symbols[j-1], symbols[j+1]
        # Check geometry first; context alone must never turn a digit into an operator.
        yes, why = slash_geometry(chars[i])
        if yes or (prev in '0123456789)' and nxt in '0123456789('):
            diag.append(f'slash candidate {i}: {yes} ({why}), context={prev}1{nxt}')
        if yes and (prev.isdigit() or prev == ')') and (nxt.isdigit() or nxt == '('):
            symbols[j] = '/'
            diag.append(f'REPLACED segment {i}: 1 -> /')

    # A slim opening parenthesis can be classified as 1, especially after /.
    for j in range(1, len(symbols)-1):
        i = keep[j]
        if symbols[j] != '1' or symbols[j-1] != '/' or flags[i] or parens[i]:
            continue
        if not symbols[j+1].isdigit():
            continue
        yes, why = v49.v48.v47.experiment.open_paren_geometry(chars[i])
        diag.append(f'opening-parenthesis candidate {i}: {yes} ({why})')
        if yes:
            symbols[j] = '('
            diag.append(f'REPLACED segment {i}: 1 -> (')
    return ''.join(symbols), diag


def segment(image):
    chars, boxes = base.segment_digits(image, method='improved', group_division=True)
    chars, boxes = base.merge_division_parts(chars, boxes)
    masked, masked_boxes, parens = base.isolate_parentheses(chars, boxes, return_flags=True)
    chars, boxes, flags = base.split_connected_digits(masked, masked_boxes, return_split_flags=True)
    by_box = {base.box_key(b): p for b, p in zip(masked_boxes, parens) if p}
    aligned = [None if f else by_box.get(base.box_key(b)) for b, f in zip(boxes, flags)]
    return chars, boxes, flags, aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-dir', type=Path, help='Optional folder of new expression PNG/JPG images')
    args = parser.parse_args()
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        raise FileNotFoundError('Missing model or original test image directory')
    model = base.load_model(str(base.MODEL))
    originals = sorted((p for p in base.IMAGES.glob('testcase*.*')
                        if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                       key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    items = [('original', p, base.EXPECTED.get(int(''.join(c for c in p.stem if c.isdigit()) or 0), ''))
             for p in originals]
    if args.new_dir:
        if not args.new_dir.is_dir():
            parser.error(f'New image folder does not exist: {args.new_dir}')
        items += [('new', p, '') for p in sorted(args.new_dir.iterdir())
                  if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    rows = []
    for group, path, expected in items:
        row = dict(group=group, image=path.name, expected=expected,
                   v49b='', v50='', changed='', diagnostics='', error='')
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot read image')
            chars, boxes, flags, parens = segment(image)
            row['v49b'], _ = v49.recognise_v49(chars, boxes, flags, model, parens)
            row['v50'], diagnostics = recognise_v50(chars, boxes, flags, model, parens)
            row['changed'] = str(row['v49b'] != row['v50'])
            row['diagnostics'] = '; '.join(diagnostics)
        except Exception as exc:
            row['error'] = repr(exc)
        rows.append(row)
        print(f"{group:8} {path.name:26} expected={expected or '?':20} "
              f"V4.9b={row['v49b']:22} V5.0={row['v50']:22} "
              f"{'ERROR: ' + row['error'] if row['error'] else row['diagnostics']}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['group','image','expected','v49b','v50',
                                                'changed','diagnostics','error'])
        writer.writeheader()
        writer.writerows(rows)
    labelled = [r for r in rows if r['expected'] and not r['error']]
    before = sum(r['v49b'] == r['expected'] for r in labelled)
    after = sum(r['v50'] == r['expected'] for r in labelled)
    regressions = [r['image'] for r in labelled if r['v49b'] == r['expected'] and r['v50'] != r['expected']]
    gains = [r['image'] for r in labelled if r['v49b'] != r['expected'] and r['v50'] == r['expected']]
    print(f'\nOriginal labelled: V4.9b {before}/{len(labelled)} | V5.0 {after}/{len(labelled)}')
    print('Newly correct:', gains, '| Regressions:', regressions)
    print('Changed:', [r['image'] for r in rows if r['changed'] == 'True'])
    print('Errors:', [(r['image'], r['error']) for r in rows if r['error']])
    print('CSV:', OUTPUT)
    print('Read-only: no working project files modified.')


if __name__ == '__main__':
    main()
