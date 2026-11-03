"""Read-only survey of disconnected shapes across HNRS testcases.

Uses the same segmentation / division merging / connected-digit splitting as V4.
Does not modify application files. Reports candidate operator shapes only.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MEMBER = ROOT / 'member_sources' / 'member_2'
sys.path.insert(0, str(MEMBER))
from Segmentation import segment_digits
from division_grouping import merge_division_parts
from connected_digit_splitter import split_connected_digits
from operator_recognition import recognise_operator

IMAGES = MEMBER / 'testimages'

def inspect(c):
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    crop = binary[y:y+h, x:x+w]
    count, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    areas = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)], reverse=True)
    mask = cv2.resize(crop, (40, 40), interpolation=cv2.INTER_NEAREST) > 0
    horizontal = float(np.mean(mask[17:23, :]))
    vertical = float(np.mean(mask[:, 17:23]))
    d1 = float(np.mean(np.diag(mask)))
    d2 = float(np.mean(np.diag(np.fliplr(mask))))
    return (w/h, count-1, areas, horizontal, vertical, d1, d2)

def main():
    paths = sorted((p for p in IMAGES.glob('testcase*.*') if p.suffix.lower() in ('.png','.jpg','.jpeg')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    print('DISCONNECTED CHARACTER SURVEY (read-only)')
    print('3-component near-square shapes and known multiplication testcases')
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()) or 0)
        image = cv2.imread(str(path))
        if image is None:
            continue
        try:
            chars, boxes = segment_digits(image, method='improved', group_division=True)
            chars, boxes = merge_division_parts(chars, boxes)
            chars, boxes, flags = split_connected_digits(chars, boxes, return_split_flags=True)
            for i, (c, flag) in enumerate(zip(chars, flags)):
                m = inspect(c)
                if m is None:
                    continue
                ratio, count, areas, horizontal, vertical, d1, d2 = m
                candidate = (count == 3 and 0.75 <= ratio <= 1.35)
                if candidate or (num in (3,4,8,11,23) and recognise_operator(c) == '*'):
                    symbol = 'CNN-only' if flag else repr(recognise_operator(c))
                    print(f'test={num:02} pos={i:02} candidate={candidate} operator={symbol:>8} '
                          f'ratio={ratio:.3f} comps={count} areas={areas} '
                          f'H={horizontal:.3f} V={vertical:.3f} D1={d1:.3f} D2={d2:.3f}')
        except Exception as exc:
            print(f'test={num:02} ERROR: {exc}')
    print('No project files modified.')

if __name__ == '__main__':
    main()
