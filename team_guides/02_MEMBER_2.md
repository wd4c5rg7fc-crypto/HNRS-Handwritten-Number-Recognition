# MEMBER 2: Segmentation
**Own:** `hnrs/segmentation.py`, tests and segmentation experiments.

1. Follow `00_READ_FIRST.md` and run the baseline tests.
2. Inspect `segment_digits(image)` in `hnrs/segmentation.py`: it preprocesses input, finds connected components, filters noise, sorts boxes left to right and returns raw crops + boxes.
3. Test quickly:
```python
from PIL import Image
import numpy as np
from hnrs.segmentation import segment_digits
crops, boxes = segment_digits(np.asarray(Image.open('assets/generated_5284.png').convert('RGB')))
print('Found:', len(crops), 'boxes:', boxes)
```
4. Compare connected components, `cv2.findContours`, and vertical projection on at least 15 images. Measure exact digit count and correct separation, not only the number of detected boxes. Record failures and screenshots.
5. Add tests for single/multiple digits, noise, empty image, touching digits, disconnected strokes. Touching digits are a known limitation: do not claim solved unless verified.
6. Keep crop output 2D uint8 and boxes `(x,y,w,h)`, ordered left-to-right. **Do not** call `prepare_digit` here; Member 4 does that after segmentation.
7. Share PR and actual comparison results. Coordinate with Member 1 on foreground convention.

**Deliver:** segmentation code, automated tests, 15-image results table, examples and limitations. **Initial handoff:** Oct 28; **final:** Oct 30.
