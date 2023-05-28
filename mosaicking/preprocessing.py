import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple
from numbers import Number


# Image Pre-processing Module
# Constant aspect ratio scaling
def const_ar_scale(img: NDArray[np.uint8], scaling: float) -> NDArray[np.uint8]:
    if scaling == 1.0:
        return img
    if scaling > 1.0:
        return cv2.resize(img, (0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_CUBIC)
    if scaling < 1.0:
        return cv2.resize(img, (0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)


def rebalance_color(img: NDArray[np.uint8], r: float, g: float, b: float) -> NDArray[np.uint8]:
    if img.ndim != 3:
        raise ValueError(f"img should have 3 dimensions, got {img.ndim}")
    return np.clip(img.astype(float)*[b, g, r], 0, 255).astype(np.uint8)


def enhance_detail(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Apply OpenCV's detailEnhance on input image.
    """
    assert img.ndim == 3, f"img should have 3 dimensions, got {img.ndim}"
    return cv2.detailEnhance(img)


def find_center(img: np.ndarray) -> Tuple[int, int]:
    """
    Extract the centroid of the image
    """
    if img.ndim > 2:
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
        x, y, w, h = cv2.boundingRect(img)
    return int(x + w * 0.5), int(y + h * 0.5)


def convex_mask(img: np.ndarray, src_pts: np.ndarray) -> NDArray[np.uint8]:
    # Create a mask based on the convex polygon of the valid keypoints in the matched area
    image_mask = np.zeros_like(img)
    poly = cv2.convexHull(src_pts).squeeze().astype(np.int32)
    cv2.fillPoly(image_mask, pts=[poly, ], color=(255, 255, 255))
    return image_mask[:, :, 0]


def crop_to_valid_area(img: np.ndarray) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    if img.ndim > 2:
        rect = cv2.boundingRect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
        rect = cv2.boundingRect(img)
    return img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], rect


def _remap_histogram(img: NDArray[np.uint8], percent: float, gamma: float) -> NDArray[np.uint8]:
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
def imadjust(img: NDArray[np.uint8], percent: Union[float, Tuple[float, ...]] = 100.0, gamma: Union[float, Tuple[float, ...]] = 1.0) -> NDArray[np.uint8]:
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
def equalize_color(img: NDArray[np.uint8], clip_limit: float = 1.0, tile_grid_size: Tuple[int, ...] = (2, 2)) -> NDArray[np.uint8]:
    """
    Given an U8 image, apply Contrast Limited Adaptive Histogram Equalization to each channel
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    output = []
    for i, channel in enumerate(cv2.split(img)):
        output.append(clahe.apply(channel))
    return cv2.merge(output)


def equalize_histogram(image: NDArray[np.uint8], clip_limit: float, tile_size: tuple) -> NDArray[np.uint8]:
    assert image.dtype == np.uint8
    # Convert the image to HSV
    clahe = cv2.createCLAHE(clip_limit, tile_size)
    # Equalize the histogram of the V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = clahe.apply(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Lighting Correction
def equalize_luminance(image: NDArray[np.uint8], clip_limit: float = 3.0, tile_grid_size: Tuple[int, ...] = (7, 7)) -> NDArray[np.uint8]:
    """
    Extract the luminance (LAB space) and apply CLAHE to it.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim < 3:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def unsharpen_image(image: NDArray[np.uint8], kernel_size: Tuple[int, ...], sigma: float) -> NDArray[np.uint8]:
    return image - cv2.GaussianBlur(image, kernel_size, sigma) + image


def sharpen_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    # Define the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Apply the kernel to the image
    result = cv2.filter2D(image, -1, kernel)

    return result
