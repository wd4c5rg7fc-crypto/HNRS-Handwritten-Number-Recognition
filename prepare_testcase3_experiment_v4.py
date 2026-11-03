"""Create an isolated V4 operator experiment and 24-image comparison.
Run from HNRS project root: python prepare_testcase3_experiment_v4.py
"""
from pathlib import Path
import re

root = Path(__file__).resolve().parent
operator_file = root / 'operator_recognition.py'
comparison_file = root / 'compare_connected_splitter_v2.py'
for path in (operator_file, comparison_file):
    if not path.is_file():
        raise FileNotFoundError(f'Required project file not found: {path.name}')
source = operator_file.read_text(encoding='utf-8')
comparison = comparison_file.read_text(encoding='utf-8')

# Insert V4 candidate rules immediately before the existing multiplication decision.
# This keeps the proven V3 multiplication and plus rules intact.
needle = re.compile(r'(?m)^(\s*)if \(\s*\n\s*not has_large_loop\s*\n\s*and \(\s*\n\s*x_multiplication')
matches = list(needle.finditer(source))
if len(matches) != 1:
    raise RuntimeError('Cannot uniquely find the V3 multiplication decision. No files changed.')
indent = matches[0].group(1)
# Indentation is 8 spaces for the if, and 8 for the candidate variable.
addition = '''        # V4 EXPERIMENT: asymmetric crossed-stroke multiplication.
        asymmetric_multiplication = (
            component_count == 1
            and 0.75 <= aspect_ratio <= 1.35
            and diagonal_2 >= 0.60
            and horizontal >= 0.50
            and vertical >= 0.40
            and not has_large_loop
        )

'''
source = source[:matches[0].start()] + addition + source[matches[0].start():]
old = '''                x_multiplication
                or asterisk_multiplication'''
new = '''                x_multiplication
                or asterisk_multiplication
                or asymmetric_multiplication'''
if source.count(old) != 1:
    raise RuntimeError('Cannot find the expected V3 multiplication alternatives. No files changed.')
source = source.replace(old, new)

# V4 disconnected plus candidate, checked in the existing plus decision.
old_plus = '''            standard_plus
            or wide_plus
            or thin_plus'''
new_plus = '''            standard_plus
            or wide_plus
            or thin_plus
            or disconnected_plus_candidate'''
if source.count(old_plus) != 1:
    raise RuntimeError('Cannot find the expected V3 plus alternatives. No files changed.')
source = source.replace(old_plus, new_plus)
plus_marker = '        if (\n            standard_plus\n'
if source.count(plus_marker) != 1:
    raise RuntimeError('Cannot uniquely locate plus decision. No files changed.')
source = source.replace(plus_marker, '''        # V4 EXPERIMENT: unusual disconnected plus from testcase3.
        # Strict conditions are provisional; validate on all images.
        disconnected_plus_candidate = (
            component_count == 2
            and 0.90 <= aspect_ratio <= 1.20
            and strongest_vertical_column >= 0.98
            and horizontal <= 0.20
            and vertical <= 0.20
            and corner_density <= 0.10
        )

''' + plus_marker)

# Sanity-check that this source is already V3, not an old backup.
if 'and vertical >= 0.55' not in source or 'and horizontal >= 0.40' not in source:
    raise RuntimeError('operator_recognition.py does not appear to contain V3 rules. No files changed.')

old_import = 'from operator_recognition import recognise_operator'
if comparison.count(old_import) != 1:
    raise RuntimeError('Cannot uniquely find the comparison script import. No files changed.')
comparison = comparison.replace(old_import, 'from operator_recognition_experiment_v4 import recognise_operator')
comparison = comparison.replace('connected_split_comparison_v2.csv', 'testcase3_operator_experiment_v4.csv')
# Ensure both baseline and split runs use V4, just as the previous experiment scripts did.
compile(source, 'operator_recognition_experiment_v4.py', 'exec')
compile(comparison, 'compare_testcase3_experiment_v4.py', 'exec')
(root / 'operator_recognition_experiment_v4.py').write_text(source, encoding='utf-8')
(root / 'compare_testcase3_experiment_v4.py').write_text(comparison, encoding='utf-8')
print('Created operator_recognition_experiment_v4.py')
print('Created compare_testcase3_experiment_v4.py')
print('Your working operator_recognition.py and app.py are unchanged.')
print('Next: python compare_testcase3_experiment_v4.py')
