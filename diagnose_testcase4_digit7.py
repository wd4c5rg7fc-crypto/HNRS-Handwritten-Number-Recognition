import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

from app import preprocess_digit

image_path = Path("test_images/gui_segments/component_7.png")
model_path = Path("models/best_cnn_model.keras")

image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(image_path)

model = load_model(str(model_path))

processed = preprocess_digit(image)
input_image = processed.reshape(1, 28, 28, 1)

probabilities = model.predict(input_image, verbose=0)[0]

print("TESTCASE4 DIGIT 7 DIAGNOSTIC")
print("Top 5 CNN predictions:")

for index in np.argsort(probabilities)[::-1][:5]:
    print(f"  Digit {index}: {probabilities[index] * 100:.2f}%")

output = Path("test_results/testcase4_digit7")
output.mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    str(output / "processed_7.png"),
    (processed * 255).astype(np.uint8)
)

print(f"\nProcessed image saved to: {output}")