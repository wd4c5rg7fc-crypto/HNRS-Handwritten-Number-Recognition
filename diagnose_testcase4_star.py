"""Inspect testcase4 multiplication symbol without modifying HNRS files.
Run immediately after recognising testcase4 in the GUI.
"""
from pathlib import Path
import cv2
import numpy as np
from operator_recognition import recognise_operator

path = Path(__file__).resolve().parent / 'test_images' / 'gui_segments' / 'component_4.png'
image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise SystemExit(f'Missing {path}\nRecognise testcase4 in the GUI first.')
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
pts = cv2.findNonZero(binary)
if pts is None:
    raise SystemExit('Empty segment')
x, y, w, h = cv2.boundingRect(pts)
crop = binary[y:y+h, x:x+w]
count, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
fg = cv2.resize(crop, (40,40), interpolation=cv2.INTER_NEAREST)>0
horizontal = float(fg[17:23,:].mean())
vertical = float(fg[:,17:23].mean())
diag1 = float(np.diag(fg).mean())
diag2 = float(np.diag(np.fliplr(fg)).mean())
corners = [fg[:10,:10],fg[:10,-10:],fg[-10:,:10],fg[-10:,-10:]]
corner_density = float(np.mean([v.mean() for v in corners]))
print('TESTCASE4 ASTERISK DIAGNOSTIC')
print('Image:', path)
print(f'crop={w}x{h}, ratio={w/h:.3f}, components={count-1}')
print(f'horizontal={horizontal:.3f}, vertical={vertical:.3f}, diag1={diag1:.3f}, diag2={diag2:.3f}, corners={corner_density:.3f}')
print('operator recognition:', repr(recognise_operator(crop)))
print('X multiplication rule:', (count-1==1 and .75<=w/h<=1.5 and diag1>=.55 and diag2>=.55 and vertical>=.45))
print('Asymmetric candidate:', (.75<=w/h<=1.35 and diag2>=.60 and horizontal>=.50 and vertical>=.40))
print('Connected component areas:', sorted([int(stats[i,cv2.CC_STAT_AREA]) for i in range(1,count)],reverse=True))
print('No project files modified.')
