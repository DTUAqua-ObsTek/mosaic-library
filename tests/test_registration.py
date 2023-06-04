import unittest
import numpy as np
from mosaicking import registration
import cv2


class TestRegistrationModule(unittest.TestCase):
    def setUp(self) -> None:
        self._blank_image = np.zeros((100, 100), np.uint8)
        # Create a black canvas
        self._orb_image = np.zeros((500, 500), dtype=np.uint8)
        # Draw some shapes on the image that are likely to produce ORB features
        cv2.circle(self._orb_image, (200, 200), 50, (255, 255, 255), -1)
        cv2.rectangle(self._orb_image, (300, 300), (400, 400), (255, 255, 255), -1)
        cv2.line(self._orb_image, (100, 400), (400, 100), (255, 255, 255), 3)
        self._detectors = cv2.SIFT_create(), cv2.ORB_create()

    def test_get_keypoints_descriptors(self):
        kp, des = registration.get_keypoints_descriptors(self._orb_image, self._detectors, None)
        kp_blank, des_blank = registration.get_keypoints_descriptors(self._blank_image, self._detectors, None)
        num_features = len(kp)
        num_features_blank = len(kp_blank)
        self.assertLess(num_features_blank, num_features)
        self.assertEqual(num_features_blank, 0)


if __name__ == '__main__':
    unittest.main()
