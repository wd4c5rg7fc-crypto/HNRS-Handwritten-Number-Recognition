"""Compare working V4 versus experimental V5 on all labelled images.

Place next to app.py, compare_connected_splitter_v2.py, and
parenthesis_mask_splitter_v5.py. Run: python compare_v5_experiment.py
Does not change app.py, Segmentation.py or operator_recognition.py.
"""
import csv
from pathlib import Path
import cv2
from tensorflow.keras.models import load_model

from compare_connected_splitter_v2 import (
    ROOT, MEMBER, MODEL, IMAGES, EXPECTED, segment_digits,
    merge_division_parts, split_connected_digits, recognise,
)
from parenthesis_mask_splitter_v5 import isolate_parentheses
from operator_recognition import recognise_operator

OUT = ROOT / 'test_results' / 'v5_comparison.csv'


def run_v5(chars, boxes, model):
    # Apply mask correction first, before existing connected-digit splitter.
    chars, boxes, parenthesis_flags = isolate_parentheses(chars, boxes, return_flags=True)
    chars, boxes, digit_flags = split_connected_digits(chars, boxes, return_split_flags=True)
    # Parenthesis flags are associated with original positions; splitting can
    # change indices, so use a separate recogniser with an aligned token stream.
    # Build original-to-split alignment using box ordering and overlap.
    from compare_connected_splitter_v2 import preprocess, calculate_expression
    import numpy as np
    symbols = []
    for char, box, digit_flag in zip(chars, boxes, digit_flags):
        parenthesis = None
        for old_box, old_flag in zip(original_boxes, original_flags):
            if old_flag and box == old_box:
                parenthesis = old_flag
                break
        if parenthesis:
            symbols.append(parenthesis)
            continue
        op = None if digit_flag else recognise_operator(char)
        if op is None:
            processed = preprocess(char)
            if processed is None:
                return '', 'EMPTY'
            probs = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)[0]
            op = str(int(np.argmax(probs)))
        symbols.append(op)
    expr = ''.join(symbols)
    try:
        answer = str(calculate_expression(expr))
    except Exception as exc:
        answer = 'ERROR: ' + str(exc)
    return expr, answer


def main():
    model = load_model(str(MODEL))
    paths = sorted((p for p in IMAGES.glob('testcase*.*') if p.suffix.lower() in ('.jpeg', '.jpg', '.png')),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    rows = []
    print('V4 vs V5 mask experiment; no project files changed')
    print(f'{"Test":<6}{"V4":<24}{"V5":<24}{"V4 OK":<9}{"V5 OK":<9}{"Masks"}')
    for path in paths:
        num = int(''.join(c for c in path.stem if c.isdigit()))
        img = cv2.imread(str(path))
        if img is None:
            continue
        try:
            chars, boxes = segment_digits(img, method='improved', group_division=True)
            chars, boxes = merge_division_parts(chars, boxes)
            base_chars, base_boxes, base_flags = split_connected_digits(chars, boxes, return_split_flags=True)
            old, _ = recognise(base_chars, model, base_flags)
            masked, _, paren_flags = isolate_parentheses(chars, boxes, return_flags=True)
            # Preserve flags in an index-safe way: use a map of the masked box
            # and character's original box, then split and classify.
            global original_boxes, original_flags
            original_boxes, original_flags = boxes, paren_flags
            new, _ = run_v5(chars, boxes, model)
            expected = EXPECTED.get(num)
            old_ok = '?' if expected is None else ('YES' if old == expected else 'NO')
            new_ok = '?' if expected is None else ('YES' if new == expected else 'NO')
            masks = sum(x is not None for x in paren_flags)
            print(f'{num:<6}{old:<24}{new:<24}{old_ok:<9}{new_ok:<9}{masks}')
            rows.append(dict(testcase=num, expected=expected or '', v4=old, v5=new,
                             v4_correct=old_ok, v5_correct=new_ok, masks=masks))
        except Exception as exc:
            print(f'{num}: ERROR {exc}')
            raise
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['testcase','expected','v4','v5','v4_correct','v5_correct','masks'])
        writer.writeheader()
        writer.writerows(rows)
    labelled = [r for r in rows if r['expected']]
    print('\nV4:', sum(r['v4_correct']=='YES' for r in labelled), '/', len(labelled))
    print('V5:', sum(r['v5_correct']=='YES' for r in labelled), '/', len(labelled))
    print('Newly correct:', [r['testcase'] for r in labelled if r['v4_correct']=='NO' and r['v5_correct']=='YES'])
    print('Regressions:', [r['testcase'] for r in labelled if r['v4_correct']=='YES' and r['v5_correct']=='NO'])
    print('CSV:', OUT)

if __name__ == '__main__':
    main()
