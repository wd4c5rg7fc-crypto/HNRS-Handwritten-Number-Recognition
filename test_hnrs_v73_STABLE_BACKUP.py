"""Compare V5.4, MathCNN, V7.1 experimental DualCNN and Hybrid on the 23 labelled images.
Run in project root: python test_hnrs_v73_all23.py
Does not alter source images or models. Does not use expected labels to choose predictions.
"""
import csv
from pathlib import Path
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import compare_four_plus_v54 as v54
from HNRS_app_v73_division_recovery import (PROJECT_DIR, MATH_CLASSES,
    preprocess_math_character, apply_mathcnn_operator_rules,
    dual_cnn_vote, recover_parentheses_v67, fuse_math_and_legacy, recover_hybrid_parentheses_v70, fuse_valid_parenthesis_disagreements_v72, recover_parenthesised_division_v73,
    merge_division_parts, isolate_parentheses, split_connected_digits,
    segment_digits, audit_digit_disagreements_v68)

MODEL_DIR = PROJECT_DIR / 'models'
OUT = PROJECT_DIR / 'test_results' / 'v73_all23_comparison.csv'
AUDIT_OUT = PROJECT_DIR / 'test_results' / 'v73_digit_disagreements.csv'

def prepare(image):
    chars, boxes = segment_digits(image, method='improved', group_division=True)
    chars, boxes = merge_division_parts(chars, boxes)
    chars, boxes, parens = isolate_parentheses(chars, boxes, return_flags=True)
    paren_by_box = {tuple(map(int, b)): flag for b, flag in zip(boxes, parens) if flag}
    chars, boxes, flags = split_connected_digits(chars, boxes, return_split_flags=True)
    paren_flags = [None if split else paren_by_box.get(tuple(map(int, box)))
                   for box, split in zip(boxes, flags)]
    return chars, boxes, flags, paren_flags

def run_one(path, digit_model, math_model):
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError('Cannot read image')
    chars, boxes, flags, parens = prepare(im)
    if not chars:
        raise ValueError('No segments')
    legacy, _ = v54.recognise_v54(chars, boxes, flags, digit_model, parens)
    inputs = [preprocess_math_character(c) for c in chars]
    if any(x is None for x in inputs):
        raise ValueError('Empty MathCNN input')
    probs = math_model.predict(np.stack(inputs)[..., None], verbose=0)
    if probs.shape[1] != len(MATH_CLASSES):
        raise ValueError(f'Unexpected MathCNN output shape: {probs.shape}')
    symbols = [p if p in ('(', ')') else MATH_CLASSES[int(np.argmax(prob))]
               for p, prob in zip(parens, probs)]
    symbols, _ = apply_mathcnn_operator_rules(chars, flags, parens, symbols)
    math_expr = ''.join(symbols)
    dual_symbols, notes = dual_cnn_vote(chars, probs, symbols, flags, parens, digit_model)
    dual_symbols, paren_notes = recover_parentheses_v67(chars, dual_symbols, probs, flags, parens)
    notes.extend(paren_notes)
    dual_expr = ''.join(dual_symbols)
    hybrid, _ = fuse_math_and_legacy(dual_symbols, probs, legacy, parens)
    hybrid, repair_notes = recover_hybrid_parentheses_v70(
        hybrid, dual_symbols, chars, probs, flags, parens)
    notes.extend(repair_notes)
    hybrid, fusion_notes = fuse_valid_parenthesis_disagreements_v72(
        hybrid, dual_symbols, chars, probs, flags, parens)
    notes.extend(fusion_notes)
    hybrid, recovery_notes = recover_parenthesised_division_v73(
        hybrid, dual_symbols, chars, probs, flags, parens)
    notes.extend(recovery_notes)
    audits = audit_digit_disagreements_v68(chars, probs, dual_symbols, flags, parens, digit_model)
    return legacy, math_expr, dual_expr, hybrid, '; '.join(notes), audits

def main():
    base = v54.base
    images = sorted((p for p in base.IMAGES.glob('testcase*.*')
                     if p.suffix.lower() in ('.jpg', '.jpeg', '.png')),
                    key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or '0'))
    if not images:
        raise FileNotFoundError(f'No testcase images in {base.IMAGES}')
    digit_model = load_model(str(MODEL_DIR / 'best_cnn_model.keras'), compile=False)
    math_model = load_model(str(MODEL_DIR / 'math_cnn_16.keras'), compile=False)
    rows = []
    audit_rows = []
    for path in images:
        number = int(''.join(filter(str.isdigit, path.stem)) or '0')
        expected = base.EXPECTED.get(number, '')
        row = dict(image=path.name, expected=expected or 'UNLABELLED',
                   v54='', mathcnn='', dualcnn='', hybrid='', vote_notes='', error='')
        try:
            (row['v54'], row['mathcnn'], row['dualcnn'],
             row['hybrid'], row['vote_notes'], audits) = run_one(path, digit_model, math_model)
            for a in audits:
                audit_rows.append(dict(image=path.name, expected=expected or 'UNLABELLED', **a))
        except Exception as exc:
            row['error'] = repr(exc)
        rows.append(row)
        print(f"{path.name:19} expected={row['expected']:22} "
              f"V5.4={row['v54']:22} MathCNN={row['mathcnn']:22} "
              f"DualCNN={row['dualcnn']:22} Hybrid={row['hybrid']:22} {row['error']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with AUDIT_OUT.open('w', newline='', encoding='utf-8') as f:
        fields = ['image', 'expected', 'segment', 'final_symbol', 'math_top', 'math_conf', 'digit_top', 'digit_conf', 'disagree', 'math_top3', 'digit_top3']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    labelled = [r for r in rows if r['expected'] != 'UNLABELLED']
    print('\nSUMMARY (exact transcription, not merely valid arithmetic):')
    print(f'Images found: {len(rows)} | Labelled: {len(labelled)}')
    for key in ('v54', 'mathcnn', 'dualcnn', 'hybrid'):
        print(f'{key:8}: {sum(r[key] == r["expected"] for r in labelled)}/{len(labelled)}')
    print('Errors:', [(r['image'], r['error']) for r in rows if r['error']])
    print('CSV:', OUT)
    print('Digit diagnostics:', AUDIT_OUT)
    print('Disagreements:', sum(bool(r['disagree']) for r in audit_rows), 'of', len(audit_rows), 'audited digit segments')
    print('Read-only: models and test images unchanged.')

if __name__ == '__main__':
    main()
