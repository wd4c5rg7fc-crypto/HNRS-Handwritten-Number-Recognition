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

output = Path("test_results/testcase4_digit7_shifts")
output.mkdir(parents=True, exist_ok=True)

print("\nTESTCASE4 DIGIT 7 SHIFT EXPERIMENT\n")

for shift in [-2, -1, 0, 1, 2]:

    matrix = np.float32([
        [1, 0, shift],
        [0, 1, 0]
    ])

    shifted = cv2.warpAffine(
        processed,
        matrix,
        (28, 28),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    probabilities = model.predict(
        shifted.reshape(1, 28, 28, 1),
        verbose=0
    )[0]

    prediction = int(np.argmax(probabilities))

    print(
        f"Shift {shift:+d}: "
        f"prediction={prediction}, "
        f"P(3)={probabilities[3]*100:.2f}%, "
        f"P(7)={probabilities[7]*100:.2f}%"
    )

    cv2.imwrite(
        str(output / f"shift_{shift:+d}.png"),
        (shifted * 255).astype(np.uint8)
    )

print(f"\nImages saved to: {output.resolve()}")