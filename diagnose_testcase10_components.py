"""Inspect testcase10's overlapping character crops without changing project code.
Run immediately after recognising testcase10 in HNRS Studio.
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'test_images' / 'gui_segments'
OUTPUT = ROOT / 'test_results' / 'testcase10_components'
OUTPUT.mkdir(parents=True, exist_ok=True)

for index, expected in [(0, '( + 8'), (4, '3 + )'), (5, '*')]:
    path = INPUT / f'component_{index}.png'
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'MISSING: {path}')
        continue
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f'\ncomponent_{index}.png (expected {expected}): size={image.shape[1]}x{image.shape[0]}')
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area < 8:
            continue
        isolated = np.zeros_like(binary)
        isolated[labels == i] = 255
        crop = isolated[y:y+h, x:x+w]
        target = OUTPUT / f'component_{index}_part_{i}.png'
        cv2.imwrite(str(target), crop)
        print(f'  part {i}: box=({x},{y},{w},{h}), area={area}, saved={target.name}')
print(f'\nIndividual components saved to: {OUTPUT}')
print('These images are for diagnosis only. No project files were changed.')
