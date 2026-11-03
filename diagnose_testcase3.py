"""Diagnose operator crops from testcase3 without changing HNRS code."""
from pathlib import Path
import cv2
import numpy as np
import operator_recognition as op

folder = Path('test_images/gui_segments')
print('TESTCASE3 OPERATOR DIAGNOSTICS')
print('Run only after testcase3 was the last GUI recognition.\n')
for index, expected in [(1, '+'), (3, '*')]:
    path = folder / f'component_{index}.png'
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'Missing {path}')
        continue
    crop = op._binary_crop(img)
    h, w = crop.shape
    resized = cv2.resize(crop, (40,40), interpolation=cv2.INTER_NEAREST)>0
    horizontal = float(np.mean(resized[17:23,:]))
    vertical = float(np.mean(resized[:,17:23]))
    corners = [resized[:10,:10],resized[:10,-10:],resized[-10:,:10],resized[-10:,-10:]]
    corner = float(np.mean([np.mean(x) for x in corners]))
    diagonal1 = float(np.mean(np.diag(resized)))
    diagonal2 = float(np.mean(np.diag(np.fliplr(resized))))
    strongest = float(np.max(np.mean(resized,axis=0)))
    n, *_ = cv2.connectedComponentsWithStats(crop,connectivity=8)
    loop = op._has_large_enclosed_loop(crop)
    asterisk = op._asterisk(crop)
    x_rule = n-1 == 1 and .75 <= w/h <= 1.50 and diagonal1 >= .55 and diagonal2 >= .55 and vertical >= .45
    standard = horizontal >= .55 and vertical >= .40 and corner <= .12
    wide = 1.20 <= w/h <= 1.80 and horizontal >= .25 and vertical >= .25 and corner <= .05
    thin = .75 <= w/h <= 1.20 and horizontal >= .40 and vertical >= .18 and strongest >= .75 and corner <= .09 and not loop
    print(f'{path.name} (expected {expected}): {w}x{h}, ratio={w/h:.3f}, components={n-1}')
    print(f'  current recogniser: {op.recognise_operator(img)!r}; loop={loop}; asterisk={asterisk}; x_rule={x_rule}')
    print(f'  horizontal={horizontal:.3f}, vertical={vertical:.3f}, corners={corner:.3f}')
    print(f'  diagonal1={diagonal1:.3f}, diagonal2={diagonal2:.3f}, strongest_column={strongest:.3f}')
    print(f'  plus rules: standard={standard}, wide={wide}, thin_v3={thin}\n')
