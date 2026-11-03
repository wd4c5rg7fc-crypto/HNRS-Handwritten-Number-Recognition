"""Diagnose testcase22 operator rules without modifying the HNRS pipeline.
Run from project root: python diagnose_testcase22.py
Run immediately after recognising testcase22 in the GUI.
"""
from pathlib import Path
import cv2
import numpy as np
import operator_recognition as op

ROOT = Path(__file__).resolve().parent
CROPS = ROOT / 'test_images' / 'gui_segments'


def inspect(filename):
    image = cv2.imread(str(CROPS / filename), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f'ERROR: Missing {CROPS / filename}')
        return
    crop = op._binary_crop(image)
    if crop is None:
        print(f'{filename}: empty')
        return
    h, w = crop.shape
    ratio = w / h
    n, _, _, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    components = n - 1
    loop = op._has_large_enclosed_loop(crop)
    star = op._asterisk(crop)
    parenthesis = op._parenthesis(crop)
    result = op.recognise_operator(image)
    print(f'\n{filename}: {w}x{h}, ratio={ratio:.3f}, components={components}')
    print(f'  operator={result!r}; parenthesis={parenthesis!r}; enclosed_loop={loop}; asterisk={star}')
    if not 0.55 <= ratio <= 1.8:
        print('  Shape rules skipped (aspect ratio outside 0.55..1.8)')
        return
    fg = cv2.resize(crop, (40, 40), interpolation=cv2.INTER_NEAREST) > 0
    horizontal = float(np.mean(fg[17:23, :]))
    vertical = float(np.mean(fg[:, 17:23]))
    corners = [fg[:10, :10], fg[:10, -10:], fg[-10:, :10], fg[-10:, -10:]]
    corner_density = float(np.mean([np.mean(c) for c in corners]))
    diag1 = float(np.mean(np.diag(fg)))
    diag2 = float(np.mean(np.diag(np.fliplr(fg))))
    strongest = float(np.max(np.mean(fg, axis=0)))
    x_mult = (components == 1 and 0.75 <= ratio <= 1.50
              and diag1 >= .55 and diag2 >= .55 and vertical >= .45)
    standard_plus = horizontal >= .55 and vertical >= .40 and corner_density <= .12
    wide_plus = (1.20 <= ratio <= 1.80 and horizontal >= .25
                 and vertical >= .25 and corner_density <= .05)
    thin_plus = (.75 <= ratio <= 1.20 and horizontal >= .47
                 and vertical >= .18 and strongest >= .75
                 and corner_density <= .09 and not loop)
    print(f'  horizontal={horizontal:.3f}; vertical={vertical:.3f}; corners={corner_density:.3f}')
    print(f'  diagonal1={diag1:.3f}; diagonal2={diag2:.3f}; strongest_column={strongest:.3f}')
    print(f'  x_multiplication={x_mult}; asterisk_multiplication={star}')
    print(f'  standard_plus={standard_plus}; wide_plus={wide_plus}; thin_plus={thin_plus}')


if __name__ == '__main__':
    print('TESTCASE22 OPERATOR DIAGNOSTICS')
    print('Only use these crops if testcase22 was the last GUI recognition.')
    for file in ('component_0.png', 'component_1.png', 'component_2.png', 'component_3.png'):
        inspect(file)
