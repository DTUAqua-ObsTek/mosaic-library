import unittest
import numpy as np
from mosaicking import preprocessing


class TestPreprocessingModule(unittest.TestCase):
    def setUp(self):
        self.img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_const_ar_scale(self):
        scaled_img = preprocessing.const_ar_scale(self.img, 0.5)
        self.assertEqual(scaled_img.shape, (50, 50, 3))

    def test_rebalance_color(self):
        rebalanced_img = preprocessing.rebalance_color(self.img, 1.0, 1.0, 1.0)
        np.testing.assert_array_equal(rebalanced_img, self.img)

    def test_find_center(self):
        center = preprocessing.find_center(self.img)
        self.assertEqual(center, (50, 50))

    def test_crop_to_valid_area(self):
        cropped_img, rect = preprocessing.crop_to_valid_area(self.img)
        self.assertEqual(cropped_img.shape, self.img.shape)
        self.assertEqual(rect, (0, 0, 100, 100))

    def test_imadjust(self):
        adjusted_img = preprocessing.imadjust(self.img, 99.0)
        self.assertEqual(adjusted_img.shape, self.img.shape)

    def test_fix_contrast(self):
        contrast_fixed_img = preprocessing.equalize_color(self.img)
        self.assertEqual(contrast_fixed_img.shape, self.img.shape)

    def test_fix_light(self):
        light_fixed_img = preprocessing.equalize_luminance(self.img)
        self.assertEqual(light_fixed_img.shape, self.img.shape)

    # Add more tests for the remaining functions in the module...


if __name__ == '__main__':
    unittest.main()

