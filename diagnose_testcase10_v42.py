"""Read-only diagnostic for testcase10 segmentation and multiplication symbol."""
from pathlib import Path
import cv2
import numpy as np
from operator_recognition import recognise_operator, _asterisk, _has_large_enclosed_loop
from parenthesis_mask_splitter_v5 import isolate_parentheses
from division_grouping import merge_division_parts
from connected_digit_splitter import split_connected_digits
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'member_sources' / 'member_2'))
from Segmentation import segment_digits

img_path = ROOT / 'member_sources' / 'member_2' / 'testimages' / 'testcase10.jpg'
if not img_path.exists():
    matches = list((ROOT / 'member_sources' / 'member_2' / 'testimages').glob('testcase10.*'))
    if not matches: raise FileNotFoundError('testcase10 image not found')
    img_path = matches[0]
image = cv2.imread(str(img_path))
if image is None: raise RuntimeError(f'Cannot read {img_path}')
chars, boxes = segment_digits(image, method='improved', group_division=True)
chars, boxes = merge_division_parts(chars, boxes)
chars, boxes, flags = split_connected_digits(chars, boxes, return_split_flags=True)
print(f'Testcase10: {len(chars)} segments')
masked, _, paren_flags = isolate_parentheses(chars, boxes, return_flags=True)
for i, (char, modified, flag) in enumerate(zip(chars, masked, paren_flags)):
    print(f'pos{i}: shape={char.shape[:2]}, operator={recognise_operator(char)!r}, parenthesis_flag={flag!r}, after_mask={recognise_operator(modified)!r}')
    if i != 5: continue
    gray = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY) if char.ndim == 3 else char
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(pts)
    crop = binary[y:y+h, x:x+w]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    mask = cv2.resize(crop, (40, 40), interpolation=cv2.INTER_NEAREST) > 0
    H = float(mask[17:23, :].mean()); V = float(mask[:, 17:23].mean())
    D1 = float(np.diag(mask).mean()); D2 = float(np.diag(np.fliplr(mask)).mean())
    corners = np.mean([mask[:10,:10].mean(), mask[:10,-10:].mean(), mask[-10:,:10].mean(), mask[-10:,-10:].mean()])
    print('  STAR METRICS:', f'ratio={w/h:.3f}', f'components={n-1}', f'areas={sorted(map(int,stats[1:,cv2.CC_STAT_AREA]), reverse=True)}', f'H={H:.3f}', f'V={V:.3f}', f'D1={D1:.3f}', f'D2={D2:.3f}', f'corners={corners:.3f}', f'asterisk={_asterisk(crop)}', f'loop={_has_large_enclosed_loop(crop)}')
print('No working project files modified.')
