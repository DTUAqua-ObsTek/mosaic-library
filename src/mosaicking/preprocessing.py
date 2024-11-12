import functools

import cv2
import numpy as np
import numpy.typing as npt
from typing import Union, Sequence, Any, Callable
from numbers import Number

from abc import ABC, abstractmethod

import mosaicking
import warnings


import logging

# Get a logger for this module
logger = logging.getLogger(__name__)

class Preprocessor(ABC):

    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def apply(self, img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat], stream: cv2.cuda.Stream = None) -> Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]:
        ...


# Image Pre-processing Module
class ColorCLAHE(Preprocessor):
    def __init__(self, clipLimit: float = 40.0, tileGridSize: tuple[int, int] = (8, 8)):
        if mosaicking.HAS_CUDA:
            self._clahe = cv2.cuda.createCLAHE(clipLimit, tileGridSize)
        else:
            self._clahe = cv2.createCLAHE(clipLimit, tileGridSize)

    def apply(self, img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat], stream: cv2.cuda.Stream = None) -> Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]:
        if mosaicking.HAS_CUDA:
            channels = cv2.cuda.split(img)
            equalized_channels = [self._clahe.apply(channel, stream, channel) for channel in channels]
            return cv2.cuda.merge(tuple(equalized_channels), img)
        channels = cv2.split(img)
        equalized_channels = [self._clahe.apply(channel) for channel in channels]
        return cv2.merge(tuple(equalized_channels), )

# Image Pre-processing Module
class Crop(Preprocessor):
    def __init__(self, roi: tuple[int, int, int, int]):
        self._roi = roi

    def apply(self, img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat], stream: cv2.cuda.Stream = None) -> Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]:
        """
        Applies a crop to a provided numpy image or GpuMat image.
        :param img: The input image.
        :type img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]
        :param stream: An optional Stream object
        :return: The cropped image.
        :rtype: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]
        """
        x, y, width, height = self._roi
        if mosaicking.HAS_CUDA and isinstance(img, cv2.cuda.GpuMat):
            return img.rowRange(y, y + height).colRange(x, x + width)
        return img[y:y+height, x:x+width]


class DistortionMapper(Preprocessor):
    #TODO: allow setting of width & height variables
    def __init__(self, K: np.ndarray, D: np.ndarray, inverse: bool = False):
        self._K = K
        self._D = D
        self._inverse = inverse


    def apply(self, image: Union[np.ndarray, cv2.cuda.GpuMat], stream: cv2.cuda.Stream = None) -> Union[np.ndarray, cv2.cuda.GpuMat]:
        """
        Applies to input image undistortion mapping or inverse mapping (distortion) if inverse flag set to True.
        """
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            width, height = image.size()
        if not self._inverse:
            xmap, ymap = cv2.initUndistortRectifyMap(self._K, self._D, None, self._K, (width, height), cv2.CV_32FC1)
        else:
            xmap, ymap = cv2.initInverseRectificationMap(self._K, self._D, np.eye(3), self._K, (width, height),
                                                                         cv2.CV_32FC1)
        if mosaicking.HAS_CUDA:
            xmap_gpu = cv2.cuda.GpuMat(xmap)
            ymap_gpu = cv2.cuda.GpuMat(ymap)
            # if input is ndarray, upload to GPU but download after
            if isinstance(image, np.ndarray):
                tmp = cv2.cuda.GpuMat(image)
                return cv2.cuda.remap(src=tmp, xmap=xmap_gpu, ymap=ymap_gpu, interpolation=cv2.INTER_CUBIC, dst=tmp, stream=stream).download()
            return cv2.cuda.remap(src=image, xmap=xmap_gpu, ymap=ymap_gpu, interpolation=cv2.INTER_CUBIC, dst=cv2.cuda.GpuMat(), stream=stream)
        return cv2.remap(image, xmap, ymap, cv2.INTER_CUBIC)


class ConstARScaling(Preprocessor):
    # TODO: Allow setting of scaling factor

    def __init__(self, scaling: float = 1.0):
        self._scaling = scaling

    def apply(self, img: npt.NDArray[np.uint8] | cv2.cuda.GpuMat, stream: cv2.cuda.Stream = None) -> npt.NDArray[np.uint8] | cv2.cuda.GpuMat:
        if self._scaling == 1.0:
            return img
        interpolant = cv2.INTER_CUBIC if self._scaling > 1.0 else cv2.INTER_AREA
        if mosaicking.HAS_CUDA:
            if isinstance(img, np.ndarray):
                tmp = cv2.cuda.GpuMat(img)
                output = cv2.cuda.resize(src=tmp, dsize=(0, 0), dst=cv2.cuda.GpuMat(), fx=self._scaling, fy=self._scaling,
                                       interpolation=interpolant, stream=stream)
                return output.download()
            return cv2.cuda.resize(img, dsize=(0, 0), dst=cv2.cuda.GpuMat(), fx=self._scaling, fy=self._scaling,
                                   interpolation=interpolant, stream=stream)
        return cv2.resize(img, (0, 0), fx=self._scaling, fy=self._scaling, interpolation=interpolant)


class Pipeline(Preprocessor):
    def __init__(self, preprocessors: Sequence[Preprocessor], args: Sequence[dict[str, Any]] = None):
        if args is None:
            args = [{}] * len(preprocessors)

        if len(preprocessors) != len(args):
            raise ValueError("The number of preprocessors must match the number of argument dictionaries.")

        # Chain each preprocessor with its corresponding arguments
        self._pipeline = functools.reduce(self.chain, zip(preprocessors, args), self.identity)

    @staticmethod
    def identity(img, **kwargs):
        return img

    @staticmethod
    def chain(f: Callable, g_arg: tuple[Preprocessor, dict[str, Any]]) -> Callable:
        g, arg = g_arg
        def chained(img, **kwargs):
            return g.apply(f(img, **kwargs), **arg)
        return chained

    def apply(self, img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat], stream: cv2.cuda.Stream = None) -> Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]:
        # TODO: maybe it's worth considering if converting back to input datatype is a good idea.
        if mosaicking.HAS_CUDA and isinstance(img, np.ndarray):
            tmp = cv2.cuda.GpuMat(img)
            output = self._pipeline(tmp, stream=stream)
        else:
            output = self._pipeline(img, stream=stream)
        if isinstance(img, np.ndarray) and isinstance(output, cv2.cuda.GpuMat):
            return output.download()
        return output


def parse_preprocessor_strings(*args: Sequence[str]) -> Sequence[Preprocessor]:
    """

    """
    mappings = {"clahe": ColorCLAHE,
                "undistort": DistortionMapper,
                "scaling": ConstARScaling,
                "crop": Crop}
    output = []
    for arg in args:
        if arg not in mappings:
            raise KeyError(f"Unknown preprocessor string {arg}, valid entries are {list(mappings.keys())}")
        output.append(mappings[arg])
    return output


# Constant aspect ratio scaling
def const_ar_scale(img: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat], scaling: float, stream: cv2.cuda.Stream = None) -> Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]:
    """
    Scale the image with constant aspect ratio restriction.
    """
    warnings.warn("const_ar_scale is deprecated and will be removed in a future release, use ConstARScaling.apply instead.", DeprecationWarning)
    return ConstARScaling(scaling=scaling).apply(img, stream=stream)


def rebalance_color(img: npt.NDArray[np.uint8], r: float, g: float, b: float) -> npt.NDArray[np.uint8]:
    """
    Apply a scaling factor to bgr color channels, clipped to [0,255]
    """
    if img.ndim != 3:
        raise ValueError(f"img should have 3 dimensions, got {img.ndim}")
    return np.clip(img.astype(float)*[b, g, r], 0, 255).astype(np.uint8)


def enhance_detail(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Apply OpenCV's detailEnhance on input image.
    """
    assert img.ndim == 3, f"img should have 3 dimensions, got {img.ndim}"
    return cv2.detailEnhance(img)


def find_center(img: np.ndarray) -> tuple[int, int]:
    """
    Extract the bounding box centroid of the non-zero component of an image
    """
    if img.ndim > 2:
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
        x, y, w, h = cv2.boundingRect(img)
    return int(x + w * 0.5), int(y + h * 0.5)


def convex_mask(img: np.ndarray, src_pts: np.ndarray) -> npt.NDArray[np.uint8]:
    # Create a mask based on the convex polygon of the valid keypoints in the matched area
    image_mask = np.zeros_like(img)
    poly = cv2.convexHull(src_pts).squeeze().astype(np.int32)
    cv2.fillPoly(image_mask, pts=[poly, ], color=(255, 255, 255))
    return image_mask[:, :, 0]


def crop_to_valid_area(img: np.ndarray) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    if img.ndim > 2:
        rect = cv2.boundingRect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
        rect = cv2.boundingRect(img)
    return img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], rect


def _remap_histogram(img: npt.NDArray[np.uint8], percent: float, gamma: float) -> npt.NDArray[np.uint8]:
    assert img.ndim == 2, f"Input image must be 2 dimensional, has {img.ndim}."
    assert 0 <= percent <= 100, f"Percentage must be between 0 and 100, got {percent}"
    quantiles = np.array([(100.0 - percent) / 200.0, 1.0 - (100.0 - percent) / 200.0])  # shave off the bottom and top percent / 2
    bounds = img.size * quantiles
    # Apply gamma correction
    img = np.clip(cv2.pow(img.astype(float) / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
    cumhist = np.cumsum(cv2.calcHist([img], [0], None, [256], (0, 256)))  # cumulative histogram
    low_cut = np.searchsorted(cumhist, bounds[0], side="left")  # find where the cumulative histogram first exceeds the lower bounds
    high_cut = np.searchsorted(cumhist, bounds[1], side="right") - 1  # find where the cumulative histogram lastly exceeds the upper bounds
    lut = np.concatenate((
        np.zeros(low_cut, dtype=np.uint8),
        np.linspace(0, 255, high_cut - low_cut + 1, dtype=np.uint8),
        255 * np.ones(255 - high_cut, dtype=np.uint8)
    ))
    return cv2.LUT(img, lut)


# Color Correction
def imadjust(img: npt.NDArray[np.uint8], percent: Union[float, tuple[float, ...]] = 100.0, gamma: Union[float, tuple[float, ...]] = 1.0) -> npt.NDArray[np.uint8]:
    """
    Apply a similar behaviour as MATLAB's imadjust, remapping the histogram truncated by upper and lower quantiles to [0, 255]
    Params:
    @param percent: total percentage of pixels to keep, imadjust will keep the pixels within the upper and lower bounds
    defined by:
    bounds = [(100 - percent) / 200, (percent) / 200]
    Best results appear at 100 or very close to 100, smaller values result in poor color balance.
    @param gamma: apply the gamma function
    """
    if issubclass(type(percent), Number):
        percent = (percent, percent, percent)
    if issubclass(type(gamma), Number):
        gamma = (gamma, gamma, gamma)
    output = img.copy()
    if img.ndim < 3:
        return _remap_histogram(img, percent[0], gamma[0])
    for i, (channel, p, g) in enumerate(zip(cv2.split(img), percent, gamma)):
        output[..., i] = _remap_histogram(channel, p, g)
    return output.astype(np.uint8)


# Contrast Equalization
def equalize_color(img: npt.NDArray[np.uint8], clip_limit: float = 1.0, tile_grid_size: tuple[int, ...] = (2, 2)) -> npt.NDArray[np.uint8]:
    """
    Given an U8 image, apply Contrast Limited Adaptive Histogram Equalization to each channel
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    output = []
    for i, channel in enumerate(cv2.split(img)):
        output.append(clahe.apply(channel))
    return cv2.merge(output)


def equalize_histogram(image: npt.NDArray[np.uint8], clip_limit: float, tile_size: tuple) -> npt.NDArray[np.uint8]:
    assert image.dtype == np.uint8
    # Convert the image to HSV
    clahe = cv2.createCLAHE(clip_limit, tile_size)
    # Equalize the histogram of the V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = clahe.apply(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Lighting Correction
def equalize_luminance(image: npt.NDArray[np.uint8], clip_limit: float = 3.0, tile_grid_size: tuple[int, ...] = (7, 7)) -> npt.NDArray[np.uint8]:
    """
    Extract the luminance (LAB space) and apply CLAHE to it.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim < 3:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def unsharpen_image(image: npt.NDArray[np.uint8], kernel_size: tuple[int, ...], sigma: float) -> npt.NDArray[np.uint8]:
    return image - cv2.GaussianBlur(image, kernel_size, sigma) + image


def sharpen_image(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    # Define the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Apply the kernel to the image
    result = cv2.filter2D(image, -1, kernel)

    return result


def make_gray(image: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]) -> Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]:
    """
    Convert an image to grayscale.
    """
    if isinstance(image, cv2.cuda.GpuMat):
        if image.channels() == 4:
            return cv2.cuda.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if image.channels() == 3:
            return cv2.cuda.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        if image.ndim > 2:
            if image.shape[-1] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.shape[-1] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def make_bgr(image: Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]) -> Union[npt.NDArray[np.uint8], cv2.cuda.GpuMat]:
    """
    Convert an image to BGR format.
    """
    if isinstance(image, cv2.cuda.GpuMat):
        if image.channels() == 4:
            return cv2.cuda.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.channels() == 2:
            return cv2.cuda.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[-1] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
