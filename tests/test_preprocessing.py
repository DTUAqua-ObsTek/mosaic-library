import unittest

import cv2
import numpy as np
from mosaicking import preprocessing
from numpy.testing import assert_array_equal


class TestPreprocessingModule(unittest.TestCase):
    def setUp(self):
        self.img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self._bup = self.img.copy()
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def test_scaling_pipeline_cls(self):
        args = [{"scaling": 0.5}, {"scaling": 0.1}, {"scaling": 5.0}]
        objs = [preprocessing.ConstARScaling(**arg) for arg in args]
        p = preprocessing.Pipeline(objs)
        scaled_img = p.apply(self.img)
        self.assertRaises(AssertionError, assert_array_equal, scaled_img, self.img)

    def test_clahe_pipeline_cls(self):
        args = [{"clipLimit": 20.0, "tileGridSize": (3, 3)}, {"clipLimit": 100.0, "tileGridSize": (8, 8)}]
        objs = [preprocessing.ColorCLAHE(**arg) for arg in args]
        p = preprocessing.Pipeline(objs)
        scaled_img = p.apply(self.img)
        self.assertRaises(AssertionError, assert_array_equal, scaled_img, self.img)
        self.assertEqual(scaled_img.shape, (100, 100, 3))

    def test_big_pipe_cls(self):
        args = [{"K": np.array([[0.5, 0.0, 50.0], [0.0, 0.5, 50.0], [0.0, 0.0, 1.0]]),
                 "D": np.array([-1e-16, 1e-16, 5e-16, -1e-16]),
                 "inverse": False},
                {"clipLimit": 20.0, "tileGridSize": (3, 3)},
                {"clipLimit": 100.0, "tileGridSize": (8, 8)},
                {"K": np.array([[0.5, 0.0, 50.0], [0.0, 0.5, 50.0], [0.0, 0.0, 1.0]]),
                 "D": np.array([-1e-2, 1e-3, 5e-4, -1e-4]),
                 "inverse": True},
                {"scaling": 0.5}
                ]
        clss = [preprocessing.DistortionMapper, preprocessing.ColorCLAHE, preprocessing.ColorCLAHE,
                preprocessing.DistortionMapper, preprocessing.ConstARScaling]
        objs = [c(**arg) for c, arg in zip(clss, args)]
        p = preprocessing.Pipeline(objs)
        output_img = p.apply(self.img)
        self.assertEqual(output_img.shape, (50, 50, 3))

    def test_const_ar_scaling_cls(self):
        scaled_img = preprocessing.ConstARScaling(0.5).apply(self.img)
        self.assertEqual(scaled_img.shape, (50, 50, 3))

    def test_const_ar_scaling_gray_cls(self):
        scaled_img = preprocessing.ConstARScaling(0.5).apply(self.gray)
        self.assertEqual(scaled_img.shape, (50, 50))

    def test_const_ar_scale(self):
        scaled_img = preprocessing.const_ar_scale(self.img, 0.5)
        self.assertEqual(scaled_img.shape, (50, 50, 3))

    def test_const_ar_scale_gray(self):
        scaled_img = preprocessing.const_ar_scale(self.gray, 0.5)
        self.assertEqual(scaled_img.shape, (50, 50))

    def test_rebalance_color(self):
        rebalanced_img = preprocessing.rebalance_color(self.img, 1.0, 1.0, 1.0)
        np.testing.assert_array_equal(rebalanced_img, self.img)

    def test_rebalance_color_gray_raises(self):
        self.assertRaises(ValueError, preprocessing.rebalance_color, self.gray, 1.0, 1.0, 1.0)

    def test_find_center(self):
        center = preprocessing.find_center(self.img)
        self.assertEqual(center, (50, 50))

    def test_find_center_gray(self):
        center = preprocessing.find_center(self.gray)
        self.assertEqual(center, (50, 50))

    def test_crop_to_valid_area(self):
        cropped_img, rect = preprocessing.crop_to_valid_area(self.img)
        self.assertEqual(cropped_img.shape, self.img.shape)
        self.assertEqual(rect, (0, 0, 100, 100))

    def test_crop_to_valid_area_gray(self):
        cropped_img, rect = preprocessing.crop_to_valid_area(self.gray)
        self.assertEqual(cropped_img.shape, self.gray.shape)
        self.assertEqual(rect, (0, 0, 100, 100))

    def test_imadjust(self):
        adjusted_img = preprocessing.imadjust(self.img, 99.0, (1.0, 0.1, 3.1))
        self.assertEqual(adjusted_img.shape, self.img.shape)

    def test_imdadjust_gray(self):
        adjusted_img = preprocessing.imadjust(self.gray, 99.0, (1.0, 0.1, 3.1))
        self.assertEqual(adjusted_img.shape, self.gray.shape)

    def test_fix_contrast(self):
        contrast_fixed_img = preprocessing.equalize_color(self.img)
        self.assertEqual(contrast_fixed_img.shape, self.img.shape)

    def test_fix_contrast_gray(self):
        contrast_fixed_img = preprocessing.equalize_color(self.gray)
        self.assertEqual(contrast_fixed_img.shape, self.gray.shape)

    def test_fix_light(self):
        light_fixed_img = preprocessing.equalize_luminance(self.img)
        self.assertEqual(light_fixed_img.shape, self.img.shape)

    def test_fix_light_gray(self):
        light_fixed_img = preprocessing.equalize_luminance(self.gray)
        self.assertEqual(light_fixed_img.shape, self.gray.shape)


if __name__ == '__main__':
    unittest.main()

