"""Read-only V4.8 vs experimental V4.9 open-zero recognition.

Place beside compare_remaining_v48.py and its dependencies.
Run: python compare_open_zero_v49.py
Only writes test_results/v49_open_zero_comparison.csv.
Never modifies the GUI, recognizer modules, source images, or models.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
import compare_remaining_v48 as v48

base = v48.base
OUT = Path(__file__).resolve().parent / 'test_results' / 'v49_open_zero_comparison.csv'


def open_zero_geometry(char):
    """Explore a near-closed oval, including handwritten zeros with a small gap.

    Uses convex-hull interior emptiness and upper-arc coverage; NOT a guarantee
    that a particular handwriting sample is zero rather than two.
    """
    m = v48.v47.experiment.mask_of(char)
    if m is None:
        return False, 'empty'
    h, w = m.shape
    aspect = w / h
    if h < 35 or not 0.70 <= aspect <= 1.30:
        return False, f'aspect={aspect:.2f}'

    ink = (m * 255).astype(np.uint8)
    points = cv2.findNonZero(ink)
    if points is None or len(points) < 100:
        return False, 'sparse'
    hull = cv2.convexHull(points)
    hull_mask = np.zeros_like(ink)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    hull_area = cv2.countNonZero(hull_mask)
    if hull_area == 0:
        return False, 'empty-hull'
    # Oval-shaped strokes surround an unfilled interior, even if a tiny gap
    # prevents cv2.RETR_CCOMP from reporting a hole.
    ink_fraction = cv2.countNonZero(ink) / hull_area
    # Measure the inner center: the open zero should not be filled there.
    inner = m[int(.32*h):max(int(.68*h), int(.32*h)+1),
              int(.30*w):max(int(.70*w), int(.30*w)+1)]
    center_fill = float(np.mean(inner)) if inner.size else 1.0
    # An upper oval arc should reach left, top and right thirds.
    upper = m[:max(1, int(.45*h))]
    top_left = np.count_nonzero(upper[:, :max(1, w//3)]) >= 8
    top_middle = np.count_nonzero(upper[:, w//3:max(w//3+1, 2*w//3)]) >= 8
    top_right = np.count_nonzero(upper[:, 2*w//3:]) >= 8
    upper_arc = top_left and top_middle and top_right
    ok = (upper_arc and 0.16 <= ink_fraction <= 0.68 and center_fill <= 0.28)
    return ok, (f'aspect={aspect:.2f} ink/hull={ink_fraction:.2f} '
                f'center-fill={center_fill:.2f} upper-arc={upper_arc}')


def recognise_v49(chars, boxes, flags, model, parens):
    """Preserve all V4.8 rules and cautiously explore a remaining 2->0."""
    old, diagnostics = v48.recognise_v48(chars, boxes, flags, model, parens)
    _, _, removed, _ = v48.v47.experiment.revised.recognise_revised(
        chars, boxes, flags, model, parens)
    keep = [i for i in range(len(chars)) if i not in removed]
    if (v48.v47.experiment.v45.edge_fragment_candidate(
            0, chars, boxes, flags, parens) and 0 in keep):
        keep.remove(0)
    if len(old) != len(keep):
        return old, diagnostics + ['V4.9 skipped: alignment mismatch']
    result = list(old)
    for j in range(1, len(result)-1):
        i = keep[j]
        # Context is a safeguard, not an expected-string or testcase-ID lookup.
        if (result[j-1] == '1' and result[j] == '2'
                and result[j+1] in ('+', '-', ')')
                and not flags[i] and not parens[i]):
            yes, why = open_zero_geometry(chars[i])
            diagnostics.append(f'open-zero candidate {i}: {yes} ({why})')
            if yes:
                result[j] = '0'
                diagnostics.append(f'REPLACED {i}: 2 -> 0')
    return ''.join(result), diagnostics


def main():
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        raise FileNotFoundError('Missing CNN model or test image directory')
    model = base.load_model(str(base.MODEL))
    paths = sorted((p for p in base.IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    rows = []
    print(f'Found {len(paths)} images. V4.8 vs experimental V4.9 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"V4.8":<23}{"V4.9":<23}Diagnostics')
    print('-'*120)
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        row = dict(testcase=num, expected=base.EXPECTED.get(num,''),
                   v48='', v49='', diagnostics='', error='')
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot read image')
            chars, boxes = base.segment_digits(image, method='improved', group_division=True)
            chars, boxes = base.merge_division_parts(chars, boxes)
            masked, masked_boxes, parens = base.isolate_parentheses(chars, boxes, return_flags=True)
            chars, boxes, flags = base.split_connected_digits(
                masked, masked_boxes, return_split_flags=True)
            by_box = {base.box_key(b): p for b,p in zip(masked_boxes, parens) if p}
            aligned = [None if f else by_box.get(base.box_key(b)) for b,f in zip(boxes, flags)]
            row['v48'], _ = v48.recognise_v48(chars, boxes, flags, model, aligned)
            row['v49'], diag = recognise_v49(chars, boxes, flags, model, aligned)
            row['diagnostics'] = '; '.join(diag)
            print(f'{num:<6}{(row["expected"] or "unknown"):<22}'
                  f'{row["v48"]:<23}{row["v49"]:<23}{row["diagnostics"]}')
        except Exception as exc:
            row['error'] = repr(exc)
            print(f'{num:<6}ERROR: {exc}')
        rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with OUT.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    labelled = [r for r in rows if r['expected'] and not r['error']]
    old = sum(r['v48'] == r['expected'] for r in labelled)
    new = sum(r['v49'] == r['expected'] for r in labelled)
    gains = [r['testcase'] for r in labelled
             if r['v48'] != r['expected'] and r['v49'] == r['expected']]
    regressions = [r['testcase'] for r in labelled
                   if r['v48'] == r['expected'] and r['v49'] != r['expected']]
    changed = [r['testcase'] for r in rows if not r['error'] and r['v48'] != r['v49']]
    print(f'\nV4.8: {old}/{len(labelled)} | V4.9: {new}/{len(labelled)}')
    print('Newly correct:', gains, '| Regressions:', regressions)
    print('Changed images (including unlabelled):', changed)
    print('Errors:', [(r['testcase'], r['error']) for r in rows if r['error']])
    print('CSV:', OUT)
    print('No working project files modified.')


if __name__ == '__main__':
    main()
