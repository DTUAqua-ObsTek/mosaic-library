import cv2
import numpy as np
from skimage.util import img_as_ubyte


# Image Pre-processing Module
# Constant aspect ratio scaling
def const_ar_scale(img, scaling):
    return cv2.resize(img, (0, 0), fx=scaling, fy=scaling)


def rebalance_color(img: np.ndarray, r: float, g: float, b: float):
    return (img.astype(float)*[b,g,r]).astype(np.uint8)


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
