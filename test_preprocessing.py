"""Explore CNN preprocessing sensitivity without changing the HNRS application.

Run from the project root:
    python test_preprocessing.py

Requires the saved GUI crop of testcase18's final digit at
    test_images/gui_segments/component_5.png
"""
from pathlib import Path
import csv
import cv2
import numpy as np
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / 'test_images' / 'gui_segments' / 'component_5.png'
MODEL_PATH = ROOT / 'models' / 'best_cnn_model.keras'
OUT_DIR = ROOT / 'test_results' / 'preprocessing_experiment'


def make_input(binary_crop, target_size=20, thickness='original', dx=0, dy=0):
    h, w = binary_crop.shape
    scale = target_size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(binary_crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized

    # Preserve the exact default app.py pipeline for the baseline.
    if thickness == 'thicker':
        canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)
    elif thickness == 'thinner':
        canvas = cv2.erode(canvas, np.ones((2, 2), np.uint8), iterations=1)

    moments = cv2.moments(canvas)
    if moments['m00']:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        shift_x = round(13.5 - cx) + dx
        shift_y = round(13.5 - cy) + dy
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, matrix, (28, 28))
    return canvas


def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f'Missing {IMAGE_PATH}\n'
            'Open testcase18 in the GUI, click Recognise, and run this script '
            'before processing another image.'
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f'Cannot read {IMAGE_PATH}')
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None:
        raise ValueError('No foreground pixels found')
    x, y, w, h = cv2.boundingRect(points)
    crop = binary[y:y + h, x:x + w]

    # A small, interpretable experiment rather than a search for a desired label.
    configs = [('baseline', 20, 'original', 0, 0)]
    configs += [(f'size_{size}', size, 'original', 0, 0) for size in (18, 19, 21, 22)]
    configs += [(f'stroke_{stroke}', 20, stroke, 0, 0) for stroke in ('thinner', 'thicker')]
    configs += [(f'shift_{direction}', 20, 'original', dx, dy)
                for direction, dx, dy in [('left', -1, 0), ('right', 1, 0),
                                          ('up', 0, -1), ('down', 0, 1)]]

    model = load_model(str(MODEL_PATH))
    rows = []
    tiles = []
    print('\nPREPROCESSING EXPERIMENT: testcase18 final digit')
    print('Baseline should match the GUI: approximately 89% for digit 5.\n')
    print(f'{"Variation":<20} {"Prediction":<12} {"P(5)":>10} {"P(6)":>10}')
    print('-' * 56)

    for name, size, thickness, dx, dy in configs:
        canvas = make_input(crop, size, thickness, dx, dy)
        input_image = (canvas.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
        probabilities = model.predict(input_image, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
        p5, p6 = float(probabilities[5]), float(probabilities[6])
        print(f'{name:<20} {prediction:<12} {p5:>9.2%} {p6:>9.2%}')
        rows.append({'variation': name, 'prediction': prediction,
                     'probability_5': round(p5, 6), 'probability_6': round(p6, 6)})
        cv2.imwrite(str(OUT_DIR / f'{name}.png'), canvas)
        large = cv2.resize(canvas, (112, 112), interpolation=cv2.INTER_NEAREST)
        tile = cv2.copyMakeBorder(large, 5, 25, 5, 5, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(tile, name, (3, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
        tiles.append(tile)

    with (OUT_DIR / 'results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # Two columns for a readable contact sheet.
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(str(OUT_DIR / 'comparison.png'), sheet)
    print(f'\nResults: {OUT_DIR / "results.csv"}')
    print(f'Images:  {OUT_DIR / "comparison.png"}')
    print('\nDo not change app.py based on this one digit alone.')


if __name__ == '__main__':
    main()
