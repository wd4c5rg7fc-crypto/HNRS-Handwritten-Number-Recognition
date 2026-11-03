"""Create an isolated testcase22 operator experiment and batch comparison.
Run from the project root: python prepare_testcase22_experiment.py
Does not modify app.py or operator_recognition.py.
"""
from pathlib import Path

root = Path(__file__).resolve().parent
operator_path = root / 'operator_recognition.py'
compare_path = root / 'compare_connected_splitter_v2.py'
for p in (operator_path, compare_path):
    if not p.is_file():
        raise FileNotFoundError(f'Missing required file: {p}')

source = operator_path.read_text(encoding='utf-8')
old_asterisk = 'asterisk_multiplication = _asterisk(crop)'
old_plus = 'and horizontal >= 0.47'
if source.count(old_asterisk) != 1 or source.count(old_plus) != 1:
    raise RuntimeError('Operator code differs from the reviewed version; no files changed.')
source = source.replace(old_asterisk, 'asterisk_multiplication = False  # experiment: disable broad asterisk heuristic')
source = source.replace(old_plus, 'and horizontal >= 0.40  # experiment: recognise weaker horizontal plus strokes')
(root / 'operator_recognition_experiment.py').write_text(source, encoding='utf-8')

comparison = compare_path.read_text(encoding='utf-8')
old_import = 'from operator_recognition import recognise_operator'
if comparison.count(old_import) != 1:
    raise RuntimeError('Comparison script differs from reviewed version; no comparison generated.')
comparison = comparison.replace(old_import, 'from operator_recognition_experiment import recognise_operator')
comparison = comparison.replace('connected_split_comparison_v2.csv', 'testcase22_operator_experiment.csv')
(root / 'compare_testcase22_experiment.py').write_text(comparison, encoding='utf-8')
print('Created operator_recognition_experiment.py')
print('Created compare_testcase22_experiment.py')
print('Original operator_recognition.py and app.py are unchanged.')
print('Next run: python compare_testcase22_experiment.py')
