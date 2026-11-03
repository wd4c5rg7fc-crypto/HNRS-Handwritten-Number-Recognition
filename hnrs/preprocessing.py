"""
Module: preprocessing.py
Author: Akarsha Thejan Wijesekara Wijesekara Liyanage (Member 1)
Project: Handwritten Number Recognition System (HNRS)
Description: Image acquisition, preprocessing, noise reduction, thresholding, normalisation, and resizing.
"""

import cv2
import numpy as np
import os

def load_image(image_path):
    """
    Method A: Load an existing image from disk in PNG, JPG, or JPEG format.
    Validates that the image can be opened and returns it as a NumPy array.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to decode image from path: {image_path}. Ensure it is a valid PNG/JPG/JPEG file.")
    
    return image

def generate_number_image(digit_folder, digit_sequence, spacing=10):
    """
    Method B: Automatically generate a multi-digit image from individual digit images stored in a folder.
    Combines them horizontally with configurable spacing and consistent height.
    Supports numbers containing 2 to 6 digits.
    """
    digit_images = []
    for char in str(digit_sequence):
        img_path = os.path.join(digit_folder, f"{char}.png")
        if not os.path.exists(img_path):
            # Fallback search for jpg
            img_path = os.path.join(digit_folder, f"{char}.jpg")
        
        if os.path.exists(img_path):
            d_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if d_img is not None:
                digit_images.append(d_img)
            else:
                raise ValueError(f"Could not read digit image for '{char}' at {img_path}")
        else:
            raise FileNotFoundError(f"Digit image for '{char}' not found in {digit_folder}")
            
    if not digit_images:
        raise ValueError("No valid digit images found for sequence generation.")
        
    # Standardise height across all digit images
    target_height = max(img.shape[0] for img in digit_images)
    standardized_digits = []
    for img in digit_images:
        if img.shape[0] != target_height:
            aspect = img.shape[1] / img.shape[0]
            new_width = int(target_height * aspect)
            img = cv2.resize(img, (new_width, target_height))
        standardized_digits.append(img)
        
    # Combine horizontally with spacing
    spacer = np.ones((target_height, spacing), dtype=standardized_digits[0].dtype) * 255
    combined_image = standardized_digits[0]
    for img in standardized_digits[1:]:
        combined_image = np.hstack((combined_image, spacer, img))
        
    return combined_image

def to_grayscale(image):
    """Convert a colour image into grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def remove_noise(image, method="gaussian"):
    """
    Apply noise reduction using Gaussian blur or Median blur.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(image, 5)
    else:
        raise ValueError("Unsupported noise reduction method. Use 'gaussian' or 'median'.")

def apply_threshold(image, method="otsu"):
    """
    Apply thresholding: global, Otsu's, or adaptive thresholding.
    Ensures final output has bright digits on a dark background (MNIST convention).
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    if method == "global":
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    elif method == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    else:
        raise ValueError("Unsupported threshold method. Use 'global', 'otsu', or 'adaptive'.")
        
    return thresh

def normalize_image(image):
    """Normalise pixel values from 0-255 range to 0.0-1.0 range."""
    return image.astype("float32") / 255.0

def resize_digit(image, size=(28, 28)):
    """
    Resize a digit while preserving its aspect ratio.
    Prevents zero-sized dimensions during resizing.
    """

    if image is None or image.size == 0:
        raise ValueError("Cannot resize an empty image.")

    h, w = image.shape[:2]
    target_h, target_w = size

    if h == 0 or w == 0:
        raise ValueError("Invalid digit dimensions.")

    # Preserve aspect ratio
    scale = min(target_h / h, target_w / w)

    # Ensure both dimensions are at least 1 pixel
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Create black canvas
    canvas = np.zeros(
        (target_h, target_w),
        dtype=image.dtype
    )

    # Centre the resized image
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2

    canvas[
        y_off:y_off + new_h,
        x_off:x_off + new_w
    ] = resized

    return canvas

def preprocess_image(image):
    """
    Preprocesses a complete multi-digit image for segmentation.
    Returns a 2D NumPy array with bright digits on a dark background, preserving layout.
    """
    if image is None:
        raise ValueError("Input image is None.")
    
    gray = to_grayscale(image)
    blurred = remove_noise(gray, method="gaussian")
    thresh = apply_threshold(blurred, method="otsu")
    
    return thresh
def prepare_digit(image):
    """Wrapper function to handle digit preparation for the app."""
    # 1. Resize to 28x28 preserving aspect ratio
    resized = resize_digit(image, size=(28, 28))
    
    # 2. Normalize pixel values to range 0.0 - 1.0
    normalized = normalize_image(resized)
    
    # 3. Add the color channel dimension so it becomes (28, 28, 1) for the CNN
    if len(normalized.shape) == 2:
        normalized = np.expand_dims(normalized, axis=-1)
        
    return normalized
