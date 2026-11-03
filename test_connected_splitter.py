"""Independent test for testcase21, without changing app.py."""
from pathlib import Path
import sys
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'member_sources' / 'member_2'))
from connected_digit_splitter import split_connected_digits

CROPS = ROOT / 'test_images' / 'gui_segments'
OUTPUT = ROOT / 'test_results' / 'connected_digits' / 'splitter_test'
OUTPUT.mkdir(parents=True, exist_ok=True)

# GUI writes these files when testcase21 is recognised. Reopen testcase21
# in the GUI first if you have since recognised a different image.
images = []
for i in range(4):
    path = CROPS / f'component_{i}.png'
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Missing {path}')
    images.append(image)

# The test uses relative boxes since saved crops have no original coordinates.
# Width/height, which drive the heuristic, are identical to the GUI crops.
boxes = [(sum(im.shape[1] for im in images[:i]), 0,
          im.shape[1], im.shape[0]) for i, im in enumerate(images)]
result, result_boxes = split_connected_digits(images, boxes)
print(f'Input components: {len(images)}')
print(f'Output components: {len(result)}')
for i, image in enumerate(result):
    path = OUTPUT / f'part_{i}.png'
    cv2.imwrite(str(path), image)
    print(f'{i}: {image.shape[1]}x{image.shape[0]} -> {path}')
print('Expected for testcase21: 5 components (3, 5, -, 1, 8).')
