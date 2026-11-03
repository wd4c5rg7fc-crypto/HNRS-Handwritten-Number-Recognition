"""Prepare a safer testcase22 operator experiment without modifying working files.
Run: python prepare_testcase22_experiment_v3.py
Then: python compare_testcase22_experiment_v3.py
"""
from pathlib import Path

root = Path(__file__).resolve().parent
operator_path = root / 'operator_recognition.py'
compare_path = root / 'compare_connected_splitter_v2.py'
for path in (operator_path, compare_path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}')

source = operator_path.read_text(encoding='utf-8')
old_asterisk = 'asterisk_multiplication = _asterisk(crop)'
new_asterisk = '''asterisk_multiplication = (
            _asterisk(crop)
            and vertical >= 0.55
        )  # experiment: require strong vertical strokes'''
old_plus = 'and horizontal >= 0.47'
new_plus = 'and horizontal >= 0.40  # experiment: accept weaker plus crossbars'
if source.count(old_asterisk) != 1 or source.count(old_plus) != 1:
    raise RuntimeError('Expected current operator rules not found exactly once. No files changed.')

comparison = compare_path.read_text(encoding='utf-8')
old_import = 'from operator_recognition import recognise_operator'
if comparison.count(old_import) != 1:
    raise RuntimeError('Expected comparison import not found exactly once. No files changed.')

modified = source.replace(old_asterisk, new_asterisk).replace(old_plus, new_plus)
compare_modified = comparison.replace(old_import, 'from operator_recognition_experiment_v3 import recognise_operator')
compare_modified = compare_modified.replace('connected_split_comparison_v2.csv', 'testcase22_operator_experiment_v3.csv')
(root / 'operator_recognition_experiment_v3.py').write_text(modified, encoding='utf-8')
(root / 'compare_testcase22_experiment_v3.py').write_text(compare_modified, encoding='utf-8')
print('Created operator_recognition_experiment_v3.py')
print('Created compare_testcase22_experiment_v3.py')
print('Your working operator_recognition.py and app.py are unchanged.')
print('Next: python compare_testcase22_experiment_v3.py')
