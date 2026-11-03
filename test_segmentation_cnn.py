import sys
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Import teammate's segmentation module
sys.path.insert(0, "member_sources/member_2")
from Segmentation import segment_digits

# Create folder for saving test images
os.makedirs("test_images", exist_ok=True)

# Load the handwritten image
image_path = "member_sources/member_2/testimages/testcase1.jpeg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Test image not found")

# Segment handwritten characters
characters, boxes = segment_digits(
    image,
    method="improved",
    group_division=True
)

print("Characters detected:", len(characters))

# Load the trained CNN model
model = load_model("models/best_cnn_model.keras")

# Known character positions in this test image:
# 1, 2, -, 5, +, 8
# Only digits are tested because the CNN recognises 0-9
digit_positions = [0, 1, 3, 5]
expected_digits = [1, 2, 5, 8]

results = []

for position, expected in zip(digit_positions, expected_digits):

    # Get the segmented character
    character = characters[position]

    # Convert to grayscale if necessary
    if len(character.shape) == 3:
        character = cv2.cvtColor(
            character,
            cv2.COLOR_BGR2GRAY
        )

    # FIX: Segmentation already returns white digits
    # on a black background, matching MNIST polarity
    _, binary = cv2.threshold(
        character,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find the white handwriting pixels
    points = cv2.findNonZero(binary)

    if points is None:
        print("No handwriting detected at position:", position)
        continue

    # Find bounding box around the digit
    x, y, w, h = cv2.boundingRect(points)

    # Crop the digit
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

    # Create a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Centre the resized digit
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = digit

    # Calculate the centre of mass
    moments = cv2.moments(canvas)

    if moments["m00"] != 0:

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # Shift the digit towards the centre
        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))

        matrix = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        canvas = cv2.warpAffine(
            canvas,
            matrix,
            (28, 28),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # Save the processed digit for inspection
    cv2.imwrite(
        f"test_images/segmented_digit_{position}.png",
        canvas
    )

    # Prepare input for CNN
    input_image = canvas.astype("float32") / 255.0
    input_image = input_image.reshape(1, 28, 28, 1)

    # Predict digit
    predictions = model.predict(
        input_image,
        verbose=0
    )[0]

    predicted = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    results.append((expected, predicted))

    print("--------------------------------")
    print("Character position:", position)
    print("Actual digit:", expected)
    print("Predicted digit:", predicted)
    print("Confidence:", round(confidence, 2), "%")
    print(
        "Result:",
        "PASS" if predicted == expected else "FAIL"
    )

# Calculate test accuracy
correct = sum(
    actual == predicted
    for actual, predicted in results
)

print("\n========== TEST SUMMARY ==========")
print("Digits tested:", len(results))
print("Correct predictions:", correct)
print("Incorrect predictions:", len(results) - correct)

if results:
    accuracy = correct / len(results) * 100
    print("Digit accuracy:", round(accuracy, 2), "%")