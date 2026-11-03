"""HNRS V5 diagnostic: isolate overlapping character components without editing project files.

Place beside app.py and run: python experiment_v5_masks.py
Run testcase10 in the GUI first, so test_images/gui_segments contains its crops.
"""
from pathlib import Path
import cv2
import numpy as np
from operator_recognition import recognise_operator

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'test_images' / 'gui_segments'
OUTPUT = ROOT / 'test_results' / 'v5_mask_experiment'
OUTPUT.mkdir(parents=True, exist_ok=True)


def components(image, min_area=8):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    result = []
    for idx in range(1, count):
        x, y, w, h, area = map(int, stats[idx])
        if area < min_area:
            continue
        isolated = np.zeros_like(binary)
        isolated[labels == idx] = 255
        result.append((isolated, (x, y, w, h, area)))
    return sorted(result, key=lambda item: item[1][0])


def crop_foreground(image):
    pts = cv2.findNonZero(image)
    if pts is None:
        return image
    x, y, w, h = cv2.boundingRect(pts)
    return image[y:y+h, x:x+w]


def main():
    print('V5 MASK DIAGNOSTIC (no changes to app.py or Segmentation.py)')
    print('Source:', SOURCE)
    for index, intended in [(0, 'opening parenthesis and 8'), (4, '3 and closing parenthesis'), (5, 'multiplication')]:
        path = SOURCE / f'component_{index}.png'
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'MISSING: {path}; run testcase10 in GUI first.')
            continue
        print(f'\ncomponent_{index}: intended {intended}; original operator={recognise_operator(img)!r}')
        parts = components(img)
        for n, (mask, box) in enumerate(parts, 1):
            cropped = crop_foreground(mask)
            out = OUTPUT / f'component_{index}_part_{n}.png'
            cv2.imwrite(str(out), cropped)
            print(f'  part {n}: box={box}, operator={recognise_operator(cropped)!r}, saved={out.name}')
        if index in (0, 4) and len(parts) == 2:
            # Diagnostic only: removing the smaller enclosed digit component from
            # the large box reveals whether the parenthesis recogniser can handle it.
            outer = max(parts, key=lambda part: part[1][4])[0]
            cleaned = crop_foreground(outer)
            out = OUTPUT / f'component_{index}_largest_only.png'
            cv2.imwrite(str(out), cleaned)
            print(f'  largest component only: operator={recognise_operator(cleaned)!r}, saved={out.name}')
    print('\nThese are diagnostic crops, not a replacement segmentation pipeline.')
    print('Please share this output and the saved *_largest_only.png images.')


if __name__ == '__main__':
    main()
