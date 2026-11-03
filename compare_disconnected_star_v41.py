"""Read-only HNRS V4 vs experimental disconnected-asterisk recognition.
Place alongside app.py and run: python compare_disconnected_star_v41.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
MEMBER = ROOT / 'member_sources' / 'member_2'
sys.path.insert(0, str(MEMBER))
from Segmentation import segment_digits
from division_grouping import merge_division_parts
from connected_digit_splitter import split_connected_digits
from operator_recognition import recognise_operator, _binary_crop, _has_large_enclosed_loop

MODEL = ROOT / 'models' / 'best_cnn_model.keras'
IMAGES = MEMBER / 'testimages'
OUTPUT = ROOT / 'test_results' / 'disconnected_star_v41_comparison.csv'
EXPECTED = {
    1:'12-5+8',2:'15+23',3:'8+3*4',4:'26-3*(17-4)',5:'18-9',
    6:'36/(3+3)',7:'24/6',8:'7*(9-4)',9:'7+5',10:'(8+3)*4',
    11:'6*4',12:'50-27',13:'15+(6*3)-4',14:'100/(5*4)',
    15:'4*(16+9)',16:'9-(2+3)',17:'(10+8)/(7-3)',18:'72/8+6',
    19:'9-(2+3)',20:'48/6+7*2',21:'35-18',22:'15+3',23:'2*3',
}

def candidate_disconnected_star(character):
    crop = _binary_crop(character)
    if crop is None:
        return False
    h, w = crop.shape
    if h == 0 or not 0.90 <= w / h <= 1.10:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n - 1 != 3:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    largest_share = float(areas.max() / areas.sum())
    if not 0.90 <= largest_share <= 0.97:
        return False
    foreground = cv2.resize(crop, (40, 40), interpolation=cv2.INTER_NEAREST) > 0
    horizontal = float(np.mean(foreground[17:23, :]))
    vertical = float(np.mean(foreground[:, 17:23]))
    return (horizontal >= 0.45 and vertical >= 0.35
            and not _has_large_enclosed_loop(crop))

def preprocess(character):
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY) if character.ndim == 3 else character.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None:
        raise ValueError('Empty character')
    x, y, w, h = cv2.boundingRect(points)
    scale = 20.0 / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(binary[y:y+h, x:x+w], (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28-nw)//2, (28-nh)//2
    canvas[oy:oy+nh, ox:ox+nw] = resized
    moments = cv2.moments(canvas)
    if moments['m00']:
        cx, cy = moments['m10']/moments['m00'], moments['m01']/moments['m00']
        dx, dy = round(13.5-cx)+1, round(13.5-cy)
        canvas = cv2.warpAffine(canvas, np.float32([[1,0,dx],[0,1,dy]]), (28,28))
    return canvas.astype(np.float32)/255.0

def compare(characters, flags, model):
    baseline, experimental, changes = [], [], []
    for index, (character, from_split) in enumerate(zip(characters, flags)):
        operator = None if from_split else recognise_operator(character)
        if operator is not None:
            original = operator
        else:
            probabilities = model.predict(preprocess(character).reshape(1,28,28,1), verbose=0)[0]
            original = str(int(np.argmax(probabilities)))
        proposed = original
        # Only consider the new rule when the existing operator detector found nothing.
        if not from_split and operator is None and candidate_disconnected_star(character):
            proposed = '*'
            changes.append(f'pos{index}:{original}->*')
        baseline.append(original)
        experimental.append(proposed)
    return ''.join(baseline), ''.join(experimental), ', '.join(changes)

def main():
    if not MODEL.is_file() or not IMAGES.is_dir():
        raise FileNotFoundError('Model or testimages directory not found')
    model = load_model(str(MODEL))
    paths = sorted((p for p in IMAGES.glob('testcase*.*')
                    if p.suffix.lower() in ('.jpg','.jpeg','.png')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    rows = []
    print(f'Found {len(paths)} images. V4 vs disconnected-star V4.1 (read-only).')
    print(f'{"Test":<6}{"Expected":<22}{"Baseline":<23}{"Experimental":<23}Changes')
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        row = dict(testcase=num, expected=EXPECTED.get(num,''), baseline='', experimental='', changes='', error='')
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot read image')
            characters, boxes = segment_digits(image, method='improved', group_division=True)
            characters, boxes = merge_division_parts(characters, boxes)
            characters, boxes, flags = split_connected_digits(characters, boxes, return_split_flags=True)
            row['baseline'], row['experimental'], row['changes'] = compare(characters, flags, model)
            print(f'{num:<6}{(row["expected"] or "unknown"):<22}{row["baseline"]:<23}{row["experimental"]:<23}{row["changes"]}')
        except Exception as exc:
            row['error'] = str(exc)
            print(f'{num:<6}ERROR: {exc}')
        rows.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['testcase','expected','baseline','experimental','changes','error'])
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if r['expected'] and not r['error']]
    base = sum(r['baseline'] == r['expected'] for r in valid)
    exp = sum(r['experimental'] == r['expected'] for r in valid)
    gains = [r['testcase'] for r in valid if r['baseline'] != r['expected'] and r['experimental'] == r['expected']]
    regressions = [r['testcase'] for r in valid if r['baseline'] == r['expected'] and r['experimental'] != r['expected']]
    print(f'\nBaseline: {base}/{len(valid)} | Experimental: {exp}/{len(valid)}')
    print(f'Newly correct: {gains} | Regressions: {regressions}')
    print(f'CSV: {OUTPUT}')
    if len(valid) != len(EXPECTED):
        print('WARNING: Some labelled testcases were not evaluated.')
    if base != 15:
        print('WARNING: Baseline differs from verified V4 15/23; investigate before accepting results.')
    print('No working project files modified.')

if __name__ == '__main__':
    main()
