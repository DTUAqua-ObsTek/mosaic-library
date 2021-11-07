import cv2
import numpy as np
from skimage.util import img_as_ubyte


# Image Pre-processing Module
# Constant aspect ratio scaling
def const_ar_scale(img, scaling):
    return cv2.resize(img, (0, 0), fx=scaling, fy=scaling)


def rebalance_color(img: np.ndarray, r: float, g: float, b: float):
    return (img.astype(float)*[b,g,r]).astype(np.uint8)


def enhance_detail(img: np.ndarray):
    return cv2.detailEnhance(img)


def find_center(img: np.ndarray):
    if img.ndim > 2:
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
        x, y, w, h = cv2.boundingRect(img)
    return int(x + w * 0.5), int(y + h * 0.5)


def convex_mask(img: np.ndarray, src_pts: np.ndarray):
    # Create a mask based on the convex polygon of the valid keypoints in the matched area
    image_mask = np.zeros_like(img)
    poly = cv2.convexHull(src_pts).squeeze().astype(np.int32)
    cv2.fillPoly(image_mask, pts=[poly, ], color=(255, 255, 255))
    return image_mask[:, :, 0]


def crop_to_valid_area(img: np.ndarray):
    if img.ndim > 2:
        rect = cv2.boundingRect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
        rect = cv2.boundingRect(img)
    return img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]], rect


# Color Correction
def fix_color(img: np.ndarray, percent: float=0.8):
    img = img_as_ubyte(img)
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut))
    return cv2.merge(out_channels).astype('uint8')


# Contrast Equalization
def fix_contrast(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img[:, :, 1] = clahe.apply(img[:, :, 1])
    img[:, :, 2] = clahe.apply(img[:, :, 2])
    return img


# Lighting Correction
def fix_light(image, limit=3, grid=(7, 7), gray=False):
    image=img_as_ubyte(image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = True
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
