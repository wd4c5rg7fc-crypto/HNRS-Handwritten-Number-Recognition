"""Read-only V4.5 vs exploratory V4.6 for handwritten + ( pairs.

Place beside compare_testcase14_v45_edge.py and its dependencies.
Run: python compare_testcase13_v46.py
No project source, trained models or GUI files are modified.
The new rules are hypotheses; review the full 24-image regression report.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
import compare_testcase14_v45_edge as v45

base = v45.base
revised = v45.revised
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'test_results' / 'testcase13_v46_comparison.csv'


def mask_of(char):
    crop = base._binary_crop(char)
    if crop is None:
        return None
    return (crop > 0).astype(np.uint8)


def plus_geometry(char):
    """Broad cross-like shape; pair context and CNN outputs provide safeguards."""
    m = mask_of(char)
    if m is None:
        return False, 'empty'
    h, w = m.shape
    if min(h, w) < 25 or not 0.65 <= w / h <= 1.4:
        return False, 'aspect'
    # Crossing strokes should span most of both axes near the central region.
    midrows = m[int(h*.30):max(int(h*.70), int(h*.30)+1)]
    midcols = m[:, int(w*.30):max(int(w*.70), int(w*.30)+1)]
    horiz = max((np.count_nonzero(row) / w for row in midrows), default=0)
    vert = max((np.count_nonzero(col) / h for col in midcols.T), default=0)
    ok = horiz >= .60 and vert >= .58
    return ok, f'cross H={horiz:.2f} V={vert:.2f}'


def open_paren_geometry(char):
    """Curved opening stroke: foreground farther right at mid-height than ends."""
    m = mask_of(char)
    if m is None:
        return False, 'empty'
    h, w = m.shape
    if h < 45 or w < 20 or not .35 <= w/h <= 1.05:
        return False, 'aspect'
    # Look at horizontal center-of-ink per vertical band, avoiding extreme edges.
    centers = []
    for lo, hi in ((.10,.25),(.42,.58),(.75,.90)):
        band = m[int(h*lo):max(int(h*hi),int(h*lo)+1)]
        ys, xs = np.nonzero(band)
        if len(xs) < 10:
            return False, 'sparse-band'
        centers.append(float(np.median(xs))/w)
    top, middle, bottom = centers
    bend = middle - (top+bottom)/2
    ok = bend >= .12 and abs(top-bottom) <= .38
    return ok, f'bend={bend:.2f} centers={[round(x,2) for x in centers]}'


def recognise_v46(chars, boxes, flags, model, parens):
    """Retain V4.5 except for a conservatively gated adjacent + ( pair."""
    _, _, removed, _ = revised.recognise_revised(chars, boxes, flags, model, parens)
    keep = [i for i in range(len(chars)) if i not in removed]
    if v45.edge_fragment_candidate(0, chars, boxes, flags, parens) and 0 in keep:
        keep.remove(0)
    # For consistency, reconstruct V4.5 text using its tested procedure.
    baseline_text, _, pre_removed, _ = revised.recognise_revised(chars, boxes, flags, model, parens)
    if v45.edge_fragment_candidate(0, chars, boxes, flags, parens) and 0 not in pre_removed:
        baseline_text = baseline_text[1:]
    text = list(baseline_text)
    diagnostics = []
    # Existing V4.5 outputs one symbol per retained character.
    if len(text) != len(keep):
        return baseline_text, ['skipped: alignment mismatch']
    for j in range(len(keep)-1):
        i, k = keep[j], keep[j+1]
        if k != i+1 or flags[i] or flags[k] or parens[i] or parens[k]:
            continue
        if text[j] != '4' or text[j+1] != '6':
            continue
        plus, pinfo = plus_geometry(chars[i])
        opening, oinfo = open_paren_geometry(chars[k])
        diagnostics.append(f'pair {i},{k}: plus={plus} ({pinfo}); paren={opening} ({oinfo})')
        if plus and opening:
            text[j], text[j+1] = '+', '('
            diagnostics.append(f'REPLACED pair {i},{k}')
    return ''.join(text), diagnostics


def main():
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        raise FileNotFoundError('Missing model or member_2/testimages')
    model = base.load_model(str(base.MODEL))
    paths = sorted((p for p in base.IMAGES.glob('testcase*.*') if p.suffix.lower() in ('.png','.jpg','.jpeg')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    print(f'Found {len(paths)} images. V4.5 vs experimental V4.6 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"V4.5":<23}{"V4.6":<23}Diagnostics')
    print('-'*120)
    rows=[]
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        row=dict(testcase=num,expected=base.EXPECTED.get(num,''),v45='',v46='',diagnostics='',error='')
        try:
            image=cv2.imread(str(path))
            if image is None: raise ValueError('Cannot read image')
            chars,boxes=base.segment_digits(image,method='improved',group_division=True)
            chars,boxes=base.merge_division_parts(chars,boxes)
            masked,masked_boxes,parens=base.isolate_parentheses(chars,boxes,return_flags=True)
            chars,boxes,flags=base.split_connected_digits(masked,masked_boxes,return_split_flags=True)
            by_box={base.box_key(b):p for b,p in zip(masked_boxes,parens) if p}
            aligned=[None if f else by_box.get(base.box_key(b)) for b,f in zip(boxes,flags)]
            old,_,removed,_=revised.recognise_revised(chars,boxes,flags,model,aligned)
            if v45.edge_fragment_candidate(0,chars,boxes,flags,aligned) and 0 not in removed:
                old=old[1:]
            row['v45']=old
            row['v46'], diag=recognise_v46(chars,boxes,flags,model,aligned)
            row['diagnostics']='; '.join(diag)
            print(f'{num:<6}{(row["expected"] or "unknown"):<22}{old:<23}{row["v46"]:<23}{row["diagnostics"]}')
        except Exception as exc:
            row['error']=repr(exc)
            print(f'{num:<6}ERROR: {exc}')
        rows.append(row)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    labelled=[r for r in rows if r['expected'] and not r['error']]
    old=sum(r['v45']==r['expected'] for r in labelled)
    new=sum(r['v46']==r['expected'] for r in labelled)
    gains=[r['testcase'] for r in labelled if r['v45']!=r['expected'] and r['v46']==r['expected']]
    regressions=[r['testcase'] for r in labelled if r['v45']==r['expected'] and r['v46']!=r['expected']]
    print(f'\nV4.5: {old}/{len(labelled)} | V4.6: {new}/{len(labelled)}')
    print('Newly correct:',gains,'| Regressions:',regressions)
    print('Errors:',[(r['testcase'],r['error']) for r in rows if r['error']])
    print('CSV:',OUTPUT)
    print('No working project files modified.')

if __name__=='__main__':
    main()
