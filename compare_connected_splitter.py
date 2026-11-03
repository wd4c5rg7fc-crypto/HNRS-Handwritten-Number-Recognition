"""Compare baseline vs connected-digit splitting across the HNRS test images.

Run from the HNRS_Combined_Project folder:
    python compare_connected_splitter.py

Read-only experiment: does not change app.py, Segmentation.py, or models.
Both paths use the same +1 pixel horizontal shift used by the GUI.
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
from operator_recognition import recognise_operator
from expression_evaluator import calculate_expression

MODEL = ROOT / 'models' / 'best_cnn_model.keras'
IMAGES = MEMBER / 'testimages'
OUTPUT = ROOT / 'test_results' / 'connected_split_comparison.csv'

# Previously confirmed labels. Testcase24 is intentionally excluded.
EXPECTED = {
    1: '12-5+8', 2: '15+23', 3: '8+3*4', 4: '26-3*(17-4)',
    5: '18-9', 6: '36/(3+3)', 7: '24/6', 8: '7*(9-4)',
    9: '7+5', 10: '(8+3)*4', 11: '6*4', 12: '50-27',
    13: '15+(6*3)-4', 14: '100/(5*4)', 15: '4*(16+9)',
    16: '9-(2+3)', 17: '(10+8)/(7-3)', 18: '72/8+6',
    19: '9-(2+3)', 20: '48/6+7*2', 21: '35-18',
    22: '15+3', 23: '2*3',
}


def preprocess(character):
    """Match current app.py preprocessing, including the +1 horizontal shift."""
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY) if character.ndim == 3 else character.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    scale = 20.0 / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(binary[y:y+h, x:x+w], (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28-nw)//2, (28-nh)//2
    canvas[oy:oy+nh, ox:ox+nw] = resized
    moments = cv2.moments(canvas)
    if moments['m00']:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        dx, dy = round(13.5-cx) + 1, round(13.5-cy)
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))
    return canvas.astype(np.float32) / 255.0


def recognise(characters, model):
    result = []
    for character in characters:
        symbol = recognise_operator(character)
        if symbol is None:
            processed = preprocess(character)
            if processed is None:
                return '', 'EMPTY CHARACTER'
            probs = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)[0]
            symbol = str(int(np.argmax(probs)))
        result.append(symbol)
    expression = ''.join(result)
    try:
        answer = str(calculate_expression(expression))
    except Exception as exc:
        answer = 'ERROR: ' + str(exc)
    return expression, answer


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    if not IMAGES.is_dir():
        raise FileNotFoundError(IMAGES)
    model = load_model(str(MODEL))
    paths = sorted(IMAGES.glob('testcase*.*'), key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    rows = []
    print(f'Found {len(paths)} images. Comparing +1px baseline vs +1px with splitting.\n')
    print(f'{"Test":<6}{"Baseline":<23}{"With splitting":<23}{"Base OK":<10}{"Split OK":<10}{"Parts"}')
    print('-' * 85)
    for path in paths:
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        number = int(''.join(c for c in path.stem if c.isdigit()))
        image = cv2.imread(str(path))
        if image is None:
            print(f'{number:<6}Could not read {path.name}')
            continue
        try:
            characters, boxes = segment_digits(image, method='improved', group_division=True)
            characters, boxes = merge_division_parts(characters, boxes)
            base_expr, base_answer = recognise(characters, model)
            split_chars, split_boxes = split_connected_digits(characters, boxes)
            split_expr, split_answer = recognise(split_chars, model)
            expected = EXPECTED.get(number)
            base_ok = 'unknown' if expected is None else ('YES' if base_expr == expected else 'NO')
            split_ok = 'unknown' if expected is None else ('YES' if split_expr == expected else 'NO')
            print(f'{number:<6}{base_expr:<23}{split_expr:<23}{base_ok:<10}{split_ok:<10}{len(characters)}->{len(split_chars)}')
            rows.append(dict(testcase=number, expected=expected or '', baseline=base_expr,
                             with_splitter=split_expr, baseline_answer=base_answer,
                             splitter_answer=split_answer, baseline_correct=base_ok,
                             splitter_correct=split_ok, original_parts=len(characters),
                             split_parts=len(split_chars)))
        except Exception as exc:
            print(f'{number:<6}ERROR: {exc}')
            rows.append(dict(testcase=number, error=str(exc)))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ['testcase','expected','baseline','with_splitter','baseline_answer','splitter_answer',
              'baseline_correct','splitter_correct','original_parts','split_parts','error']
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    labelled = [r for r in rows if r.get('baseline_correct') in ('YES','NO')]
    base = sum(r['baseline_correct'] == 'YES' for r in labelled)
    split = sum(r['splitter_correct'] == 'YES' for r in labelled)
    regressions = [r['testcase'] for r in labelled if r['baseline_correct']=='YES' and r['splitter_correct']!='YES']
    gains = [r['testcase'] for r in labelled if r['baseline_correct']=='NO' and r['splitter_correct']=='YES']
    print(f'\nVerified expression accuracy (of {len(labelled)} labelled images):')
    print(f'  Baseline (+1px): {base}/{len(labelled)}')
    print(f'  With splitter:  {split}/{len(labelled)}')
    print(f'  Newly correct:  {gains}')
    print(f'  Regressions:    {regressions}')
    print(f'CSV: {OUTPUT}')
    print('Note: correct means the entire recognised expression matches its label.')


if __name__ == '__main__':
    main()
