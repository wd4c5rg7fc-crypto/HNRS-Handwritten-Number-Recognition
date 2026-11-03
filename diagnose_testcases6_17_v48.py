"""Read-only V4.8 diagnostics for remaining testcases 6 and 17.

Place beside compare_testcase15_v47.py and its dependencies.
Run: python diagnose_testcases6_17_v48.py
No GUI, model, source image or recognition module is changed.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
import compare_testcase15_v47 as v47

base = v47.base
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'test_results' / 'v48_remaining_diagnostics.csv'


def describe(char):
    m = v47.experiment.mask_of(char)
    if m is None:
        return 'empty'
    h, w = m.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = sorted((int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, n)), reverse=True)
    return f'{w}x{h} ratio={w/max(h,1):.2f} components={n-1} areas={areas[:5]}'


def main():
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        raise FileNotFoundError('Model or test images directory missing')
    model = base.load_model(str(base.MODEL))
    paths = sorted((p for p in base.IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    rows = []
    total = 0
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        if num not in (6, 17):
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read {path}')
        chars, boxes = base.segment_digits(image, method='improved', group_division=True)
        before_merge = len(chars)
        chars, boxes = base.merge_division_parts(chars, boxes)
        before_parens = len(chars)
        masked, masked_boxes, parens = base.isolate_parentheses(chars, boxes, return_flags=True)
        chars, boxes, flags = base.split_connected_digits(masked, masked_boxes, return_split_flags=True)
        by_box = {base.box_key(b): p for b, p in zip(masked_boxes, parens) if p}
        aligned = [None if f else by_box.get(base.box_key(b)) for b, f in zip(boxes, flags)]
        old, diag = v47.recognise_v47(chars, boxes, flags, model, aligned)
        _, _, removed, _ = v47.experiment.revised.recognise_revised(chars, boxes, flags, model, aligned)
        keep = [i for i in range(len(chars)) if i not in removed]
        if v47.experiment.v45.edge_fragment_candidate(0, chars, boxes, flags, aligned) and 0 in keep:
            keep.remove(0)
        print(f'\nTESTCASE {num} | expected={base.EXPECTED.get(num)} | V4.7={old}')
        print(f'Segments: initial={before_merge}, after division merge={before_parens}, final={len(chars)}')
        print('V4.7 diagnostics:', '; '.join(diag) or '(none)')
        if len(old) != len(keep):
            print(f'WARNING: output/segment mismatch: {len(old)} symbols, {len(keep)} kept segments')
        for position, i in enumerate(keep):
            box = tuple(map(int, boxes[i]))
            symbol = old[position] if position < len(old) else '?'
            star, sinfo = v47.star_geometry(chars[i])
            closing, cinfo = v47.closing_paren_geometry(chars[i])
            plus, pinfo = v47.experiment.plus_geometry(chars[i])
            opening, oinfo = v47.open_paren_geometry_revised(chars[i])
            info = describe(chars[i])
            print(f'  position={position:02d} segment={i:02d} output={symbol!r} '
                  f'box={box} split={flags[i]} paren={aligned[i]!r} {info}')
            print(f'    shape candidates: star={star} ({sinfo}); close={closing} ({cinfo}); '
                  f'plus={plus} ({pinfo}); open={opening} ({oinfo})')
            rows.append(dict(testcase=num, position=position, segment=i, output=symbol,
                             box=str(box), split=flags[i], paren=aligned[i] or '',
                             shape=info, star=star, star_details=sinfo,
                             closing=closing, closing_details=cinfo,
                             plus=plus, plus_details=pinfo,
                             opening=opening, opening_details=oinfo))
        total += 1
    if total != 2:
        print(f'WARNING: found {total} of 2 requested testcases')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['testcase'])
        writer.writeheader()
        writer.writerows(rows)
    print('\nDiagnostics CSV:', OUT)
    print('No working project files modified.')


if __name__ == '__main__':
    main()
