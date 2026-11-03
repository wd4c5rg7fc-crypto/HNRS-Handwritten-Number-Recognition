"""Read-only V4.7 vs exploratory V4.8 on all 24 expression images.

Save in the project root beside compare_testcase15_v47.py and dependencies.
Run: python compare_remaining_v48.py
Only writes test_results/v48_remaining_comparison.csv.
Rules are experimental and MUST pass the full regression suite before GUI use.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
import compare_testcase15_v47 as v47

base = v47.base
OUT = Path(__file__).resolve().parent / 'test_results' / 'v48_remaining_comparison.csv'


def zero_loop(char):
    """Require an enclosed interior hole, rather than guessing 0 from CNN's 2."""
    m = v47.experiment.mask_of(char)
    if m is None:
        return False, 'empty'
    h, w = m.shape
    if h < 35 or not .55 <= w / h <= 1.35:
        return False, f'aspect={w/h:.2f}'
    contours, hierarchy = cv2.findContours((m * 255), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False, 'no-contours'
    holes = [cv2.contourArea(c) for i, c in enumerate(contours) if hierarchy[0][i][3] >= 0]
    largest = max(holes, default=0)
    return largest >= .035 * w * h, f'largest-hole={largest:.0f} ({largest/(w*h):.1%})'


def division_fragments(first, second, first_box, second_box):
    """Wide upper stroke plus tiny lower dot, horizontally aligned, vertically separated."""
    a = v47.experiment.mask_of(first)
    b = v47.experiment.mask_of(second)
    if a is None or b is None:
        return False, 'empty'
    x1, y1, w1, h1 = map(int, first_box)
    x2, y2, w2, h2 = map(int, second_box)
    top, bottom = sorted(((x1,y1,w1,h1,a), (x2,y2,w2,h2,b)), key=lambda z:z[1])
    tx,ty,tw,th,tm = top
    bx,by,bw,bh,bm = bottom
    overlap = max(0, min(tx+tw,bx+bw)-max(tx,bx))
    gap = by-(ty+th)
    area_ratio = np.count_nonzero(bm)/max(1,np.count_nonzero(tm))
    ok = (tw >= 45 and tw/th >= 1.45 and bw <= .40*tw
          and bh <= .65*th and .03 <= area_ratio <= .30
          and overlap >= .60*bw and 0 <= gap <= 1.7*th)
    return ok, f'upper-ratio={tw/th:.2f} lower/upper-area={area_ratio:.2f} gap={gap} overlap={overlap}'


def recognise_v48(chars, boxes, flags, model, parens):
    old, old_diag = v47.recognise_v47(chars, boxes, flags, model, parens)
    _, _, removed, _ = v47.experiment.revised.recognise_revised(chars, boxes, flags, model, parens)
    keep = [i for i in range(len(chars)) if i not in removed]
    if v47.experiment.v45.edge_fragment_candidate(0, chars, boxes, flags, parens) and 0 in keep:
        keep.remove(0)
    diag = list(old_diag)
    if len(old) != len(keep):
        return old, diag + ['alignment mismatch']
    symbols = list(old)

    # Fix '+' misclassified as 1, only between operands and with crossing strokes.
    for j in range(1, len(symbols)-1):
        i = keep[j]
        if (symbols[j] == '1' and symbols[j-1].isdigit() and symbols[j+1].isdigit()
                and not flags[i] and not parens[i]):
            yes, why = v47.experiment.plus_geometry(chars[i])
            diag.append(f'plus candidate {i}: {yes} ({why})')
            if yes:
                symbols[j] = '+'
                diag.append(f'REPLACED {i}: 1 -> +')

    # Fix CNN 2 -> 0 when a genuine enclosed loop is present.
    for j in range(1, len(symbols)-1):
        i = keep[j]
        if (symbols[j] == '2' and symbols[j-1] == '1'
                and symbols[j+1] in ('+', '-', ')')
                and not flags[i] and not parens[i]):
            yes, why = zero_loop(chars[i])
            diag.append(f'zero candidate {i}: {yes} ({why})')
            if yes:
                symbols[j] = '0'
                diag.append(f'REPLACED {i}: 2 -> 0')

    # Fix closing ')' before division when a '(' is still open.
    for j in range(1, len(symbols)-1):
        i = keep[j]
        if (symbols[j] == '2' and symbols[j+1] == '/'
                and symbols[:j].count('(') > symbols[:j].count(')')
                and not flags[i] and not parens[i]):
            yes, why = v47.closing_paren_geometry(chars[i])
            diag.append(f'middle closing candidate {i}: {yes} ({why})')
            if yes:
                symbols[j] = ')'
                diag.append(f'REPLACED {i}: 2 -> )')

    # Fix final ')' predicted 3, only if an unmatched '(' remains.
    if symbols and symbols[-1] == '3' and symbols.count('(') > symbols.count(')'):
        i = keep[-1]
        if not flags[i] and not parens[i]:
            yes, why = v47.closing_paren_geometry(chars[i])
            diag.append(f'final closing candidate {i}: {yes} ({why})')
            if yes:
                symbols[-1] = ')'
                diag.append(f'REPLACED {i}: 3 -> )')

    # Combine a segmented horizontal division bar and a small lower dot.
    # Must occur between a number and an opening parenthesis.
    for j in range(len(symbols)-2, 0, -1):
        if (symbols[j:j+3] == ['2', '*', '('] and symbols[j-1].isdigit()
                and keep[j+1] == keep[j]+1
                and not flags[keep[j]] and not flags[keep[j+1]]
                and not parens[keep[j]] and not parens[keep[j+1]]):
            yes, why = division_fragments(chars[keep[j]], chars[keep[j+1]],
                                          boxes[keep[j]], boxes[keep[j+1]])
            diag.append(f'division pair {keep[j]},{keep[j+1]}: {yes} ({why})')
            if yes:
                symbols[j:j+2] = ['/']
                diag.append(f'MERGED {keep[j]},{keep[j+1]}: 2* -> /')
    return ''.join(symbols), diag


def main():
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        raise FileNotFoundError('Missing model or test image directory')
    model = base.load_model(str(base.MODEL))
    paths = sorted((p for p in base.IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    rows = []
    print(f'Found {len(paths)} images. V4.7 vs experimental V4.8 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"V4.7":<23}{"V4.8":<23}Diagnostics')
    print('-'*120)
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        row = dict(testcase=num, expected=base.EXPECTED.get(num,''),
                   v47='', v48='', diagnostics='', error='')
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot read image')
            chars, boxes = base.segment_digits(image, method='improved', group_division=True)
            chars, boxes = base.merge_division_parts(chars, boxes)
            masked, masked_boxes, parens = base.isolate_parentheses(chars, boxes, return_flags=True)
            chars, boxes, flags = base.split_connected_digits(masked, masked_boxes, return_split_flags=True)
            by_box = {base.box_key(b): p for b,p in zip(masked_boxes,parens) if p}
            aligned = [None if f else by_box.get(base.box_key(b)) for b,f in zip(boxes,flags)]
            row['v47'], _ = v47.recognise_v47(chars,boxes,flags,model,aligned)
            row['v48'], diag = recognise_v48(chars,boxes,flags,model,aligned)
            row['diagnostics'] = '; '.join(diag)
            print(f'{num:<6}{(row["expected"] or "unknown"):<22}{row["v47"]:<23}{row["v48"]:<23}{row["diagnostics"]}')
        except Exception as exc:
            row['error'] = repr(exc)
            print(f'{num:<6}ERROR: {exc}')
        rows.append(row)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labelled = [r for r in rows if r['expected'] and not r['error']]
    old = sum(r['v47'] == r['expected'] for r in labelled)
    new = sum(r['v48'] == r['expected'] for r in labelled)
    gains = [r['testcase'] for r in labelled if r['v47'] != r['expected'] and r['v48'] == r['expected']]
    regressions = [r['testcase'] for r in labelled if r['v47'] == r['expected'] and r['v48'] != r['expected']]
    changed = [r['testcase'] for r in rows if r['v47'] != r['v48'] and not r['error']]
    print(f'\nV4.7: {old}/{len(labelled)} | V4.8: {new}/{len(labelled)}')
    print('Newly correct:',gains,'| Regressions:',regressions)
    print('Changed images (including unlabelled):',changed)
    print('Errors:',[(r['testcase'],r['error']) for r in rows if r['error']])
    print('CSV:',OUT)
    print('No working project files modified.')

if __name__ == '__main__':
    main()
