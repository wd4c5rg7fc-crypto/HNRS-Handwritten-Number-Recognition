import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained CNN.
model = load_model("models/best_cnn_model.keras")

# Load the final character from testcase18.
image = cv2.imread(
    "test_images/gui_segments/component_5.png",
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError("Could not load component_5.png")

# Convert to binary.
_, binary = cv2.threshold(
    image,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Crop around the handwritten digit.
points = cv2.findNonZero(binary)

if points is None:
    raise ValueError("No foreground pixels found")

x, y, w, h = cv2.boundingRect(points)
crop = binary[y:y+h, x:x+w]

# Resize while preserving aspect ratio.
scale = 20 / max(w, h)

new_w = max(1, round(w * scale))
new_h = max(1, round(h * scale))

resized = cv2.resize(
    crop,
    (new_w, new_h),
    interpolation=cv2.INTER_AREA
)

# Centre on a 28x28 canvas.
canvas = np.zeros((28, 28), dtype=np.uint8)

start_x = (28 - new_w) // 2
start_y = (28 - new_h) // 2

canvas[
    start_y:start_y+new_h,
    start_x:start_x+new_w
] = resized

# Centre using image moments.
moments = cv2.moments(canvas)

if moments["m00"] > 0:
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    shift_x = int(round(13.5 - cx))
    shift_y = int(round(13.5 - cy))

    matrix = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    canvas = cv2.warpAffine(
        canvas,
        matrix,
        (28, 28)
    )

# Normalise for the CNN.
input_image = canvas.astype(np.float32) / 255.0
input_image = input_image.reshape(1, 28, 28, 1)

# Predict all digit probabilities.
predictions = model.predict(
    input_image,
    verbose=0
)[0]

print("\nCNN DIGIT PROBABILITIES")
print("=======================")

for digit in np.argsort(predictions)[::-1]:
    print(f"Digit {digit}: {predictions[digit] * 100:.2f}%")

print("\nPredicted digit:", int(np.argmax(predictions)))

# Save the actual 28x28 input for visual inspection.
cv2.imwrite(
    "test_results/debug_digit18_preprocessed.png",
    canvas
)

print("Saved preprocessed image.")