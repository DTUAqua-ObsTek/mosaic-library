import cv2
import numpy as np
from typing import Union, Tuple, List, Sequence
from numpy import typing as npt
from itertools import chain
import mosaicking
from mosaicking.preprocessing import make_gray
from abc import ABC, abstractmethod





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

    @abstractmethod
    def detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None) -> Tuple[Tuple[cv2.KeyPoint, ...], npt.NDArray[np.float32]]:
        ...

class OrbDetector(FeatureDetector):

    def __init__(self):
        self._detector = cv2.cuda.ORB.create() if mosaicking.HAS_CUDA else cv2.ORB.create()

    def detect(self, img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat], mask: npt.NDArray[np.uint8] = None, stream: cv2.cuda.Stream = None) -> Tuple[Tuple[cv2.KeyPoint, ...], npt.NDArray[np.float32]]:
        return self._cuda_detect(img, mask) if mosaicking.HAS_CUDA else self._cpu_detect(img, mask)

    def _cuda_detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None ) -> Tuple[Tuple[cv2.KeyPoint, ...], npt.NDArray[np.float32]]:
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
        kp, desc = self._detector.detectAndComputeAsync(gpu, gpu_mask)
        if kp.download() is None:
            return tuple(), np.empty((0, 32), dtype=np.uint8)
        kp = cv2.KeyPoint.convert(kp.download()[:2, :].T)
        return kp, desc.download()

    def _cpu_detect(self, img, mask=None):
        return self._detector.detectAndCompute(make_gray(img), mask)


class SurfDetector(FeatureDetector):

    def __init__(self, hessian_threshold: float = 0.5):
        self._detector = cv2.cuda.SURF_CUDA.create(hessian_threshold) if mosaicking.HAS_CUDA else cv2.xfeatures2d.SURF.create(hessian_threshold)

    def detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None, stream: cv2.cuda.Stream = None) -> Tuple[Tuple[cv2.KeyPoint, ...], npt.NDArray[np.float32]]:
        return self._cuda_detect(img, mask) if mosaicking.HAS_CUDA else self._cpu_detect(img, mask)

    def _cuda_detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None ) -> Tuple[Tuple[cv2.KeyPoint, ...], npt.NDArray[np.float32]]:
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

    def _cpu_detect(self, img, mask=None):
        return self._detector.detectAndCompute(make_gray(img), mask)


class SiftDetector(FeatureDetector):

    def __init__(self):
        self._detector = cv2.SIFT.create()

    def detect(self, img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat],
               mask: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat] = None) -> Tuple[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]:
        if isinstance(img, cv2.cuda.GpuMat):
            img = img.download()
        if isinstance(mask, cv2.cuda.GpuMat):
            mask = mask.download()
        return self._detector.detectAndCompute(img, mask, None, False)


class BriskDetector(FeatureDetector):

    def __init__(self):
        self._detector = cv2.BRISK.create()

    def detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None) -> Tuple[Tuple[cv2.KeyPoint, ...], npt.NDArray[np.float32]]:
        return self._detector.detectAndCompute(img, mask, None, False)


class KazeDetector(FeatureDetector):

    def __init__(self):
        self._detector = cv2.KAZE.create()

    def detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None) -> Tuple[Tuple[cv2.KeyPoint, ...], npt.NDArray[np.float32]]:
        return self._detector.detectAndCompute(img, mask, None, False)


class AkazeDetector(FeatureDetector):

    def __init__(self):
        self._detector = cv2.AKAZE.create()

    def detect(self, img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] = None) -> Tuple[Tuple[cv2.KeyPoint, ...], npt.NDArray[np.float32]]:
        return self._detector.detectAndCompute(img, mask, None, False)


class Matcher:
    def __init__(self):
        self._matcher = cv2.cuda.DescriptorMatcher.createBFMatcher(cv2.NORM_L2) if mosaicking.HAS_CUDA else cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    def knn_match(self, query: np.ndarray, train: np.ndarray) -> Sequence[Sequence[cv2.DMatch]]:
        if mosaicking.HAS_CUDA:
            gpu_query = cv2.cuda.GpuMat()
            gpu_query.upload(query.astype(np.float32))
            gpu_train = cv2.cuda.GpuMat()
            gpu_train.upload(train.astype(np.float32))
            return self._matcher.knnMatch(gpu_query, gpu_train, k=2)
        return self._matcher.knnMatch(query, train, k=2)

    def match(self, query: np.ndarray, train: np.ndarray) -> Sequence[cv2.DMatch]:
        if mosaicking.HAS_CUDA:
            gpu_query = cv2.cuda.GpuMat()
            gpu_query.upload(query.astype(np.float32))
            gpu_train = cv2.cuda.GpuMat()
            gpu_train.upload(train.astype(np.float32))
            return self._matcher.match(gpu_query, gpu_train)
        return self._matcher.match(query, train)


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
