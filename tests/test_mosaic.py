import tempfile
import unittest

import mosaicking.transformations
from mosaicking import mosaic
import mosaicking
import numpy as np
from scipy.spatial.transform import Rotation
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.DEBUG)

class TestMosaic(unittest.TestCase):

    def setUp(self):
        data_path = Path(__file__).parent / 'data' / "sparus_snippet.mp4"
        output_path = Path(tempfile.mkdtemp())
        orientation_path = Path(__file__).parent / 'data' / "sparus_snippet.csv"
        orientation_time_offset_path = Path(__file__).parent / 'data' / "sparus_time_offset.yaml"
        with open(orientation_time_offset_path, "r") as f:
            orientation_time_offset = yaml.safe_load(f)["video"]
        K = np.array(
            [405.6384738851233, 0, 189.9054317917407, 0, 405.588335378204, 139.9149578253755, 0, 0, 1]).reshape((3, 3))
        D = np.array([-0.3670656233416921, 0.203001968694465, 0.003336917744124004, -0.000487426354679637, 0])
        self._preprocessors = (("clahe", dict(), None),)
        self._mosaic = mosaic.SequentialMosaic(data_path=data_path,
                                project_path=output_path,
                                feature_types=("ORB",),  # , "SIFT"),
                                intrinsics={"K": K, "D": D},
                                orientation_path=orientation_path,
                                orientation_time_offset=orientation_time_offset,
                                force_cpu=True,
                                keypoint_roi=True,
                                bovw_clusters=5,
                                alpha=1.0)

    def tearDown(self):
        # Remove the temporary directory and database
        db_dir = self._mosaic._model.graph["db_dir"]
        for child in db_dir.iterdir():
            child.unlink()
        db_dir.rmdir()

    def test_mosaic_save(self):
        self._mosaic.save()
        test_mosaic = mosaic.SequentialMosaic.load(self._mosaic._project_path)

    def test_mosaic_feature_extraction(self):
        self._mosaic.extract_features()

    def test_mosaic_matching(self):
        self._mosaic.extract_features()
        self._mosaic.match_features()

    def test_mosaic_registration(self):
        self._mosaic.extract_features()
        self._mosaic.match_features()
        self._mosaic.registration()

    def test_mosaic_global(self):
        self._mosaic.extract_features()
        self._mosaic.match_features()
        self._mosaic.registration()
        self._mosaic.global_registration()

    def test_mosaic_tile(self):
        self._mosaic.extract_features()
        self._mosaic.match_features()
        self._mosaic.registration()
        self._mosaic.global_registration()
        self._mosaic.generate((4096, 4096))

    def test_bovw(self):
        self._mosaic.extract_features()
        self._mosaic.global_features()
        self._mosaic.node_knn(0)

    def test_mosaic_preprocessing(self):
        self._mosaic._preprocessor_pipeline, self._mosaic._preprocessor_args = self._mosaic._create_preprocessor_obj(self._preprocessors)
        self._mosaic.extract_features()

    def test_corner_warp(self):
        R = Rotation.from_euler('ZYX', (90, 0, 0), degrees=True)
        K = np.array([[1, 0, 50],
                     [0, 1, 25],
                      [0, 0, 1]], float)
        H = K @ R.as_matrix() @ mosaicking.transformations.inverse_K(K)
        H = np.stack((H, np.linalg.inv(H)), axis=0)
        crns = mosaic.get_corners(H, 100, 50)
        H = H[0]
        crns_1 = mosaic.get_corners(H, 100, 50)



if __name__ == '__main__':
    unittest.main()
