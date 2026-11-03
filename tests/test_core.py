import unittest
import numpy as np
from hnrs.core import preprocess_digit, segment_digits

class CoreTests(unittest.TestCase):
    def test_blank(self):
        result = preprocess_digit(np.full((50, 50), 255, dtype=np.uint8))
        self.assertEqual(result.shape, (28, 28, 1))
        self.assertEqual(float(result.max()), 0)

    def test_two_components(self):
        image = np.full((70, 140), 255, dtype=np.uint8)
        image[10:55, 15:28] = 0
        image[10:55, 80:95] = 0
        digits, boxes = segment_digits(image)
        self.assertEqual(len(digits), 2)
        self.assertLess(boxes[0][0], boxes[1][0])

if __name__ == '__main__':
    unittest.main()
