import warnings

import cv2
import numpy as np
from typing import Union, Tuple, List, Sequence, Any
from numpy import typing as npt
from itertools import chain
import mosaicking
from mosaicking.preprocessing import make_gray
from abc import ABC, abstractmethod
from mosaicking.core import concatenate_with_slices


def get_keypoints_descriptors(img: npt.NDArray[np.uint8], detectors: Sequence[cv2.Feature2D], mask: npt.NDArray[np.uint8] = None) -> Tuple[Tuple[cv2.KeyPoint], Tuple[Union[npt.NDArray[np.float32], npt.NDArray[np.uint8]]]]:
    features = [get_features(img, detector, mask) for detector in detectors]
    kp = tuple(chain.from_iterable([f[0] for f in features]))
    des = tuple(f[1] for f in features if f[1] is not None)
    return kp, des


def get_match_points(kp_src: Sequence[cv2.KeyPoint], kp_dst: Sequence[cv2.KeyPoint], matches: Sequence[cv2.DMatch]) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Converts matched keypoint objects from source and destination into numpy arrays.
    """
    assert max([m.queryIdx for m in matches]) < len(kp_src), "Match indices are larger than length of src keypoints, check order of src, dst keypoint arguments."
    assert max([m.trainIdx for m in matches]) < len(kp_dst), "Match indices are larger than length of dst keypoints, check order of src, dst keypoint arguments."
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts


class FeatureDetector(ABC):

    def __init__(self, *args, **kwargs) -> None:
        self._detector = self._create(*args, **kwargs)


    @property
    @abstractmethod
    def feature_type(self) -> str:
        ...

    @abstractmethod
    def _create(self, *args, **kwargs) -> Union[cv2.Feature2D, Sequence[cv2.Feature2D], cv2.cuda.SURF_CUDA, 'cv2.cuda.Feature2DAsync']:
        """Initialize the detector here."""
        ...

    def detect(self, img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat],
               mask: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat] = None) -> tuple[
        Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]:
        if isinstance(img, cv2.cuda.GpuMat):
            img = img.download()
        if isinstance(mask, cv2.cuda.GpuMat):
            mask = mask.download()
        return self._detector.detectAndCompute(img, mask, None, False)


class OrbDetector(FeatureDetector):

    def _create(self, force_cpu: bool = False, *args, **kwargs) -> Union[cv2.ORB, 'cv2.cuda.ORB']:
        self._flag = force_cpu
        return cv2.cuda.ORB.create() if mosaicking.HAS_CUDA and not self._flag else cv2.ORB.create()

    def detect(self, img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat], mask: npt.NDArray[np.uint8] = None, stream: cv2.cuda.Stream = None) -> tuple[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]:
        if mosaicking.HAS_CUDA and not self._flag:
            return self._cuda_detect(img, mask)
        if isinstance(img, cv2.cuda.GpuMat):
            return self._cpu_detect(img.download(), mask)
        return self._cpu_detect(img, mask)

    @property
    def feature_type(self):
        return "ORB"

    def _cuda_detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None) -> tuple[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]:
        if isinstance(img, np.ndarray):
            gpu = cv2.cuda.GpuMat(make_gray(img))
        else:
            gpu = make_gray(img)
        gpu_mask = cv2.cuda.GpuMat(mask)
        if mask is not None:
            gpu_mask = make_gray(gpu_mask)
        # TODO: some detectors using pyramid scaling (SIFT, ORB) will fail here if the image size is too small.
        kp, desc = self._detector.detectAndComputeAsync(gpu, gpu_mask)
        if kp.download() is None:
            return tuple(), np.empty((0, 32), dtype=np.uint8)
        kp = cv2.KeyPoint.convert(kp.download()[:2, :].T)
        return kp, desc.download()

    def _cpu_detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None) -> tuple[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]:
        return self._detector.detectAndCompute(make_gray(img), mask)


class SurfDetector(FeatureDetector):

    def _create(self, hessian_threshold: float = 0.5, *args, **kwargs) -> Union[cv2.xfeatures2d.SURF, cv2.cuda.SURF_CUDA]:
        return cv2.cuda.SURF_CUDA.create(hessian_threshold) if mosaicking.HAS_CUDA else cv2.xfeatures2d.SURF.create(hessian_threshold)

    def detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None, stream: cv2.cuda.Stream = None) -> tuple[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]:
        return self._cuda_detect(img, mask) if mosaicking.HAS_CUDA else self._cpu_detect(img, mask)

    @property
    def feature_type(self):
        return "SURF"

    def _cuda_detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None ) -> tuple[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]:
        if isinstance(img, cv2.cuda.GpuMat):
            gpu = cv2.cuda.GpuMat(make_gray(img))
        else:
            gpu = cv2.cuda.GpuMat()
            gpu.upload(make_gray(img))
        gpu_mask = cv2.cuda.GpuMat()
        if mask is not None:
            gpu_mask.upload(make_gray(mask))
        else:
            gpu_mask.upload(None)
        kp, desc = self._detector.detectWithDescriptors(gpu, gpu_mask, None, None, False)
        if kp.download() is None:
            return tuple(), np.empty((0, 32), dtype=np.uint8)
        kp = cv2.KeyPoint.convert(kp.download()[:2, :].T)
        return kp, desc.download()

    def _cpu_detect(self, img, mask=None) -> tuple[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]:
        return self._detector.detectAndCompute(make_gray(img), mask)


class SiftDetector(FeatureDetector):

    @property
    def feature_type(self):
        return "SIFT"

    def _create(self, *args, **kwargs) -> cv2.SIFT:
        return cv2.SIFT.create()


class BriskDetector(FeatureDetector):

    @property
    def feature_type(self):
        return "BRISK"

    def _create(self, *args, **kwargs) -> cv2.BRISK:
        return cv2.BRISK.create()


class KazeDetector(FeatureDetector):

    @property
    def feature_type(self):
        return "KAZE"

    def _create(self, *args, **kwargs) -> cv2.KAZE:
        return cv2.KAZE.create()


class AkazeDetector(FeatureDetector):

    @property
    def feature_type(self):
        return "AKAZE"

    def _create(self, *args, **kwargs) -> cv2.AKAZE:
        return cv2.AKAZE.create()


def parse_detectors(feature_types: Sequence[str], *args, **kwargs) -> Sequence[FeatureDetector]:
    """
    Map strings to a feature detector constructor.
    """
    mappings = {"ORB": OrbDetector,
                "SIFT": SiftDetector,
                "SURF": SurfDetector,
                "KAZE": KazeDetector,
                "AKAZE": AkazeDetector,
                "BRISK": BriskDetector,
                }
    if "ALL" in feature_types:
        return tuple(m(*args, **kwargs) for m in mappings.values())
    return tuple(mappings[f](*args, **kwargs) for f in feature_types)


class CompositeDetector(FeatureDetector):

    def _create(self, feature_types: Sequence[str], *args, **kwargs) -> Sequence[FeatureDetector]:
        return parse_detectors(feature_types, *args, **kwargs)

    def detect(self, img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat], mask: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat] = None) -> dict[str, dict[str, Union[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]]]:
        features = dict()
        for detector in self._detector:
            kp, desc = detector.detect(img, mask)
            features.update({detector.feature_type: {"keypoints": kp,
                                             "descriptors": desc}})
        return features

    def feature_type(self) -> Sequence[str]:
        return tuple(d.feature_type for d in self._detector)


class Matcher:
    def __init__(self):
        self._matcher = cv2.cuda.DescriptorMatcher.createBFMatcher(cv2.NORM_L2) if mosaicking.HAS_CUDA else cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    def knn_match(self, query: np.ndarray, train: np.ndarray, mask: np.ndarray = None) -> Sequence[Sequence[cv2.DMatch]]:
        if mosaicking.HAS_CUDA:
            gpu_query = cv2.cuda.GpuMat()
            gpu_query.upload(query.astype(np.float32))
            gpu_train = cv2.cuda.GpuMat()
            gpu_train.upload(train.astype(np.float32))
            gpu_mask = cv2.cuda.GpuMat()
            gpu_mask.upload(mask)
            return self._matcher.knnMatch(gpu_query, gpu_train, k=2, mask=gpu_mask)
        return self._matcher.knnMatch(query, train, k=2, mask=mask)

    def match(self, query: np.ndarray, train: np.ndarray, mask: np.ndarray = None) -> Sequence[cv2.DMatch]:
        if mosaicking.HAS_CUDA:
            gpu_query = cv2.cuda.GpuMat()
            gpu_query.upload(query.astype(np.float32))
            gpu_train = cv2.cuda.GpuMat()
            gpu_train.upload(train.astype(np.float32))
            gpu_mask = cv2.cuda.GpuMat()
            gpu_mask.upload(mask)
            return self._matcher.match(gpu_query, gpu_train, gpu_mask)
        return self._matcher.match(query, train, mask)


class CompositeMatcher(Matcher):
    """
    Class to perform feature matching on several feature sets.
    """
    def _get_combined(self, query: Sequence[np.ndarray], train: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert len(query) == len(train), "query and train feature sets should be same length."
        # Only match descriptors with like descriptors, mask should be setup correctly
        q, q_slices = concatenate_with_slices(query, 0)
        t, t_slices = concatenate_with_slices(train, 0)
        mask = np.zeros((q.shape[0], t.shape[0]), dtype=np.uint8)
        for q_slice, t_slice in zip(q_slices, t_slices):
            mask[q_slice, t_slice] = 1
        return q, t, mask

    def knn_match(self, query: dict[str, npt.NDArray[np.float32]], train: dict[str, npt.NDArray[np.float32]], mask: np.ndarray = None) -> dict[str, Sequence[Sequence[cv2.DMatch]]]:
        if mask is not None:
            warnings.warn("mask variable unused.")
        matches = dict()
        feature_types = query.keys()
        for feature_type in feature_types:
            if query[feature_type] is None or train[feature_type] is None:
                matches.update({feature_type: tuple()})
                continue
            matches.update({feature_type: super().knn_match(query[feature_type], train[feature_type])})
        return matches

    def match(self, query: dict[str, npt.NDArray[np.float32]], train: dict[str, npt.NDArray[np.float32]], mask: np.ndarray = None) -> dict[str, Sequence[cv2.DMatch]]:
        if mask is not None:
            warnings.warn("mask variable unused.")
        matches = dict()
        for feature_type in query:
            matches.update({feature_type: super().match(query[feature_type], train[feature_type])})
        return matches


def get_matches(descriptors1: Union[npt.NDArray[float], Sequence[npt.NDArray[float]]], descriptors2: Union[npt.NDArray[float], Sequence[npt.NDArray[float]]], matcher: Matcher, minmatches: int):
    if not isinstance(descriptors1, (tuple, list)):
        descriptors1 = [descriptors1]
    if not isinstance(descriptors2, (tuple, list)):
        descriptors2 = [descriptors2]
    mlength = 0
    nlength = 0
    good = []
    for d1, d2 in zip(descriptors1, descriptors2):
        matches = matcher.match(d1.astype(np.float32), d2.astype(np.float32))
        for m, n in matches:
            m.queryIdx = m.queryIdx + mlength
            m.trainIdx = m.trainIdx + nlength
            n.queryIdx = n.queryIdx + nlength
            n.trainIdx = n.trainIdx + mlength
            if m.distance < 0.7 * n.distance:
                good.append(m)
        mlength += d1.shape[0]
        nlength += d2.shape[0]
    return minmatches <= len(good), good

def get_features(img: npt.NDArray[np.uint8], fdet: FeatureDetector, mask: npt.NDArray[np.uint8] = None) -> Tuple[List[cv2.KeyPoint], Union[npt.NDArray[np.float32], npt.NDArray[np.uint8]]]:
    """
    Given a feature detector, obtain the keypoints and descriptors found in the image.
    """
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return fdet.detect(img, mask)



