"""Preprocessing, digit segmentation and image composition for HNRS."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def foreground(gray):
    """Return white handwriting on a black background as uint8 image."""
    gray = np.asarray(gray, dtype=np.uint8)
    # Determine polarity from border brightness, robust to dark/light backgrounds.
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    if np.median(border) > 127:
        gray = 255 - gray
    return gray


def preprocess_digit(gray):
    """MNIST-style centering and normalization, output (28,28,1) float32."""
    gray = foreground(gray)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((28, 28, 1), dtype=np.float32)
    crop = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape
    factor = 20.0 / max(h, w)
    resized = cv2.resize(crop, (max(1, round(w*factor)), max(1, round(h*factor))), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    rh, rw = resized.shape
    y, x = (28-rh)//2, (28-rw)//2
    canvas[y:y+rh, x:x+rw] = resized
    moments = cv2.moments(canvas)
    if moments['m00']:
        cx, cy = moments['m10']/moments['m00'], moments['m01']/moments['m00']
        matrix = np.float32([[1, 0, 13.5-cx], [0, 1, 13.5-cy]])
        canvas = cv2.warpAffine(canvas, matrix, (28, 28), flags=cv2.INTER_LINEAR)
    return (canvas.astype(np.float32)/255.0)[..., None]


def segment_digits(image):
    """Segment left-to-right digits using connected components.

    Limitation: touching digits can merge; disconnected strokes can split.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    fg = foreground(gray)
    _, binary = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    height, width = gray.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < max(5, height*width*0.0001) or h < 3 or w < 1:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda b: b[0])
    digits = []
    for x, y, w, h in boxes:
        pad = max(2, int(0.08*max(w,h)))
        roi = fg[max(0,y-pad):min(height,y+h+pad), max(0,x-pad):min(width,x+w+pad)]
        digits.append(preprocess_digit(roi))
    return digits, boxes


def compose_number(folder, number, spacing=8):
    """Compose digit files named 0.png..9.png into a multi-digit image."""
    if not number or not number.isdigit():
        raise ValueError('Enter a nonempty sequence of digits.')
    images = []
    for char in number:
        path = Path(folder) / (char + '.png')
        if not path.exists():
            raise FileNotFoundError(f'Missing digit image: {path}')
        images.append(Image.open(path).convert('L'))
    height = max(im.height for im in images) + 20
    width = sum(im.width for im in images) + spacing*(len(images)-1) + 20
    result = Image.new('L', (width, height), 255)
    x = 10
    for im in images:
        result.paste(im, (x, (height-im.height)//2))
        x += im.width + spacing
    return result
