import unittest
from mosaicking import core
import cv2
import numpy as np


class TestCore(unittest.TestCase):
    def test_imagenode_dataclass_asdict(self):
        features = {"test_feature": {"keypoints": [cv2.KeyPoint(x=0.1, y=0.1, size=10.0),
                                  cv2.KeyPoint(x=0.2, y=0.2, size=20.0)],
                    "descriptors": np.random.randint(255, size=(2, 128), dtype=np.uint8),}}
        cls = core.ImageNode(features=features)
        self.assertTrue(isinstance(cls.features["test_feature"]["keypoints"], np.ndarray))

if __name__ == '__main__':
    unittest.main()
