import unittest
import numpy as np
from hnrs.preprocessing import preprocess_image, prepare_digit
from hnrs.segmentation import segment_digits
from hnrs.image_acquisition import generate_number_image

class TestTeamInterfaces(unittest.TestCase):
    def test_preprocess_and_segmentation(self):
        image = np.full((70,160,3),255,np.uint8)
        image[12:55,15:33]=0
        image[12:55,100:125]=0
        binary = preprocess_image(image)
        self.assertEqual(binary.shape,(70,160))
        self.assertEqual(binary.dtype,np.uint8)
        crops, boxes = segment_digits(image)
        self.assertEqual(len(crops),2)
        self.assertEqual(len(boxes),2)
        self.assertLess(boxes[0][0],boxes[1][0])
        for crop in crops:
            tensor=prepare_digit(crop)
            self.assertEqual(tensor.shape,(28,28,1))
            self.assertTrue(0<=tensor.min()<=tensor.max()<=1)

if __name__=='__main__': unittest.main()
