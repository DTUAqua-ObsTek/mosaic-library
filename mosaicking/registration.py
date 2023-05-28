import cv2
import numpy as np
from typing import Union, Tuple, List, Sequence
from numpy import typing as npt


def get_matches(descriptors1: Union[npt.NDArray[float], list], descriptors2: Union[npt.NDArray[float], list], matcher: cv2.DescriptorMatcher, minmatches: int):
    if not isinstance(descriptors1, list):
        descriptors1 = [descriptors1]
    if not isinstance(descriptors2, list):
        descriptors2 = [descriptors2]
    mlength = 0
    nlength = 0
    good = []
    for d1, d2 in zip(descriptors1, descriptors2):
        matches = matcher.knnMatch(d1.astype(np.float32), d2.astype(np.float32), 2)
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


def get_features(img: npt.NDArray[np.uint8], fdet: cv2.Feature2D, mask: npt.NDArray[np.uint8] = None) -> Tuple[List[cv2.KeyPoint], npt.NDArray[float]]:
    """
    Given a feature detector, obtain the features found in the image.
    """
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return fdet.detectAndCompute(img, mask)


def get_match_points(kp_src: Sequence[cv2.KeyPoint], kp_dst: Sequence[cv2.KeyPoint], matches: Sequence[cv2.DMatch]) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Converts matched keypoint objects from source and destination into numpy arrays.
    """
    assert max([m.queryIdx for m in matches]) < len(kp_src), "Match indices are larger than length of src keypoints, check order of src, dst keypoint arguments."
    assert max([m.trainIdx for m in matches]) < len(kp_dst), "Match indices are larger than length of dst keypoints, check order of src, dst keypoint arguments."
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts
