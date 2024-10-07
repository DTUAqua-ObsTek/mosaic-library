import unittest
import mosaicking
from mosaicking import mosaic, core
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

class TestMosaic(unittest.TestCase):

    # TODO: test preprocessor pipeline

    def setUp(self):
        K = np.array([[543.3327734182214, 0.0, 489.02536042247897],
                        [0.0, 542.398772982566, 305.38727712002805],
                        [0.0, 0.0, 1.0]])
        D = np.array([-0.1255945656257394, 0.053221287232781606, 9.94070021080493e-05,
    9.550660927242349e-05])
        self._preprocessors = (("undistort", {"K": K, "D": D}, None),
                               ("scaling", {"scaling": 0.5}, None),
                               ("clahe", {"clipLimit": 100.0, "tileGridSize": (11, 11)}, None),)
        self._mosaic = mosaic.SequentialMosaic(data_path="Data/export_harbor_sequence_2.mp4",
                                               output_path="test_sequential/",
                                               feature_types=("ORB",), #, "SIFT"),
                                               intrinsics={"K": K, "D": D},
                                               orientation_path="Data/tf_cam_ned_harbor_sequence_2.csv",
                                               time_offset=1523958586.902587138,
                                               verbose=False,
                                               caching=False,
                                               force_cpu=True,
                                               player_params={"finish": 500})

    def test_mosaic_save(self):
        self._mosaic.save()

    def test_mosaic_feature_extraction(self):
        self._mosaic.extract_features()

    def test_mosaic_matching(self):
        self._mosaic.extract_features()
        self._mosaic.match_features()

    def test_mosaic_registration(self):
        self._mosaic.extract_features()
        self._mosaic.match_features()
        self._mosaic.registration()

    def test_mosaic_preprocessing(self):
        self._mosaic._preprocessor_pipeline, self._mosaic._preprocessor_args = self._mosaic._create_preprocessor_obj(self._preprocessors)
        self._mosaic.extract_features()

    def test_mosaic_global(self):
        self._mosaic.extract_features()
        self._mosaic.match_features()
        self._mosaic.registration()
        self._mosaic.global_registration()

    def test_mosaic_tile(self):
        self._mosaic.extract_features()
        self._mosaic.match_features()
        self._mosaic.registration()
        # TODO: something strange hapening here at frame 327
        self._mosaic.global_registration()
        self._mosaic.generate((1024, 1024))

    def test_save_graph(self):
        self._mosaic.extract_features()
        self._mosaic.match_features()
        self._mosaic.registration()
        self._mosaic.global_registration()
        with open("/tmp/graph.pkl", "wb") as f:
            pickle.dump(self._mosaic._registration_obj, f)

    def test_corner_warp(self):
        R = Rotation.from_euler('ZYX', (90, 0, 0), degrees=True)
        K = np.array([[1, 0, 50],
                     [0, 1, 25],
                      [0, 0, 1]], float)
        H = K @ R.as_matrix() @ core.inverse_K(K)
        H = np.stack((H, np.linalg.inv(H)), axis=0)
        crns = mosaic.get_corners(H, 100, 50)
        H = H[0]
        crns_1 = mosaic.get_corners(H, 100, 50)



if __name__ == '__main__':
    unittest.main()
