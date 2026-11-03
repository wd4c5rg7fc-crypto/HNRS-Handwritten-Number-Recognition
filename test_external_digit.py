import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the original handwritten digit image
img = cv2.imread("test_images/digit7.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Handwritten digit image not found")

# Convert to white handwriting on a black background
_, binary = cv2.threshold(
    img,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Save thresholded image for Week 9 evidence
cv2.imwrite("test_images/digit7_threshold.png", binary)

# Find the bounding box around the handwriting
points = cv2.findNonZero(binary)

if points is None:
    raise ValueError("No handwritten digit detected")

x, y, w, h = cv2.boundingRect(points)

# Crop the handwritten digit
digit = binary[y:y+h, x:x+w]

# Resize while preserving aspect ratio
scale = 20 / max(w, h)

new_w = max(1, round(w * scale))
new_h = max(1, round(h * scale))

digit = cv2.resize(
    digit,
    (new_w, new_h),
    interpolation=cv2.INTER_AREA
)

# Centre the digit on a 28x28 black canvas
canvas = np.zeros((28, 28), dtype=np.uint8)

x_offset = (28 - new_w) // 2
y_offset = (28 - new_h) // 2

canvas[
    y_offset:y_offset + new_h,
    x_offset:x_offset + new_w
] = digit

# Save final preprocessed image for evidence
cv2.imwrite("test_images/digit7_final.png", canvas)

# Normalise pixel values to the range 0-1
input_image = canvas.astype("float32") / 255.0

# Reshape image into the CNN input format
input_image = input_image.reshape(1, 28, 28, 1)

print("Preprocessed image shape:", input_image.shape)

# Load the trained CNN model
model = load_model("models/best_cnn_model.keras")

# Predict the handwritten digit
predictions = model.predict(input_image, verbose=0)[0]

predicted_digit = int(np.argmax(predictions))
confidence = float(np.max(predictions)) * 100

# Display prediction results
print("--------------------------------")
print("External Handwritten Digit Test")
print("--------------------------------")
print("Actual digit: 5")
print("Predicted digit:", predicted_digit)
print("Confidence:", round(confidence, 2), "%")

# Check whether prediction is correct
if predicted_digit == 5:
    print("Test Result: PASS")
else:
    print("Test Result: FAIL")

print("Thresholded image saved successfully")
print("Final preprocessed image saved successfully")