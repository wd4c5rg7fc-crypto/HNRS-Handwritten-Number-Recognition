"""Evaluate untouched handwritten test images using the same Hybrid V7.3 pipeline.
Images must be newly handwritten, not generated from test/development examples.
Usage: python evaluate_fresh_v73.py --images fresh_handwriting
"""
import argparse, csv, sys
from pathlib import Path
from tensorflow.keras.models import load_model
from test_hnrs_v73_all23 import run_one
from HNRS_app_v73_division_recovery import PROJECT_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, default=Path('fresh_handwriting'))
    ap.add_argument('--manifest', type=Path, default=Path('fresh_test_manifest.csv'))
    ap.add_argument('--output', type=Path, default=Path('test_results/v73_fresh_holdout.csv'))
    args = ap.parse_args()
    with args.manifest.open(newline='', encoding='utf-8-sig') as f:
        cases = list(csv.DictReader(f))
    if not cases:
        raise SystemExit('Manifest empty')
    names = [r['filename'] for r in cases]
    if len(set(names)) != len(names):
        raise SystemExit('Duplicate filenames in manifest')
    missing = [n for n in names if not (args.images/n).is_file()]
    if missing:
        print('Missing images (do not score incomplete set):', ', '.join(missing))
        return 2
    model_dir = PROJECT_DIR/'models'
    digit = load_model(str(model_dir/'best_cnn_model.keras'), compile=False)
    math = load_model(str(model_dir/'math_cnn_16.keras'), compile=False)
    from expression_evaluator import calculate_expression
    rows=[]
    for case in cases:
        result = dict(case, prediction='', predicted_answer='', transcription_correct=False,
                      answer_correct=False, error='')
        try:
            _, _, _, expression, _, _ = run_one(args.images/case['filename'], digit, math)
            result['prediction'] = expression
            result['transcription_correct'] = expression == case['expected_expression']
            try:
                answer = calculate_expression(expression)
                result['predicted_answer'] = str(answer)
                result['answer_correct'] = abs(float(answer)-float(case['expected_answer'])) < 1e-8
            except Exception as exc:
                result['error'] = f'evaluation: {exc}'
        except Exception as exc:
            result['error'] = f'recognition: {exc}'
        rows.append(result)
        print(f"{case['filename']:16} expected={case['expected_expression']:22} "
              f"predicted={result['prediction']:22} exact={result['transcription_correct']} "
              f"answer={result['predicted_answer']} {result['error']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    total=len(rows)
    exact=sum(r['transcription_correct'] for r in rows)
    answer=sum(r['answer_correct'] for r in rows)
    errors=sum(bool(r['error']) for r in rows)
    print(f'\nUNSEEN HOLDOUT: exact transcription {exact}/{total} ({100*exact/total:.1f}%)')
    print(f'Answer correct: {answer}/{total} ({100*answer/total:.1f}%) | Errors: {errors}')
    print(f'CSV: {args.output.resolve()}')
    return 0 if exact == total else 1

if __name__ == '__main__':
    sys.exit(main())
