"""
Module: test_preprocessing.py
Author: Akarsha Thejan Wijesekara Wijesekara Liyanage (Member 1)
Project: Handwritten Number Recognition System (HNRS)
Description: Automated unit tests for image acquisition and preprocessing functions.
"""

import numpy as np
import pytest
from preprocessing import to_grayscale, remove_noise, apply_threshold, normalize_image, resize_digit

def test_to_grayscale():
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray_img = to_grayscale(color_img)
    assert len(gray_img.shape) == 2
    assert gray_img.shape == (100, 100)

def test_remove_noise():
    img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    filtered = remove_noise(img, method="gaussian")
    assert filtered.shape == (50, 50)

def test_apply_threshold():
    img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    thresh = apply_threshold(img, method="otsu")
    assert thresh.shape == (50, 50)
    assert thresh.dtype == np.uint8

def test_normalize_image():
    img = np.array([[0, 128, 255]], dtype=np.uint8)
    norm = normalize_image(img)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0

def test_resize_digit():
    img = np.random.randint(0, 255, (40, 60), dtype=np.uint8)
    resized = resize_digit(img, size=(28, 28))
    assert resized.shape == (28, 28)
