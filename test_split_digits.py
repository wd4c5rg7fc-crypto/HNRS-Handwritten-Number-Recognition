"""Test CNN recognition of the two split crops from testcase21.

Place alongside app.py and run: python test_split_digits.py
"""
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from app import preprocess_digit

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'best_cnn_model.keras'
IMAGE_DIR = ROOT / 'test_results' / 'connected_digits'

model = load_model(str(MODEL_PATH))
print('\nCONNECTED DIGIT CNN TEST')
print('========================')

for filename in ('left.png', 'right.png'):
    image_path = IMAGE_DIR / filename
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Cannot read {image_path}')
    processed = preprocess_digit(image)
    if processed is None:
        raise ValueError(f'Cannot preprocess {filename}')
    probabilities = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)[0]
    print(f'\n{filename}')
    for digit in np.argsort(probabilities)[::-1][:3]:
        print(f'Digit {digit}: {probabilities[digit] * 100:.2f}%')
    print('Prediction:', int(np.argmax(probabilities)))
