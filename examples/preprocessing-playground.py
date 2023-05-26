import argparse
import cv2
from mosaicking import preprocessing
from pathlib import Path
from numpy import typing as npt
import numpy as np
from scipy.interpolate import interp1d


class HSVGUI:
    def __init__(self):
        self._clip_limit = 1.0
        self._tile_size = (5, 5)
        self._enable = False
        cv2.createTrackbar("Toggle Value", "display", 0, 1, self._toggle)
        cv2.createTrackbar("V_clip", "display", 100, 1000, self._clip_limit_callback)
        cv2.createTrackbar("V_tile", "display", 5, 15, self._tile_size_callback)

    def _toggle(self, val):
        self._enable = val

    def _clip_limit_callback(self, val: int):
        self._clip_limit = float(val) / 100.0

    def _tile_size_callback(self, val: int):
        self._tile_size = (val + 1, val + 1)

    def __call__(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return preprocessing.equalize_histogram(img, self._clip_limit, self._tile_size) if self._enable else img


class LuminanceGUI:
    def __init__(self):
        self._clip_limit = 1.0
        self._tile_size = (5, 5)
        self._enable = False
        cv2.createTrackbar("Toggle Luminance", "display", 0, 1, self._toggle)
        cv2.createTrackbar("L_clip", "display", 100, 1000, self._clip_limit_callback)
        cv2.createTrackbar("L_tile", "display", 5, 15, self._tile_size_callback)

    def _toggle(self, val):
        self._enable = val

    def _clip_limit_callback(self, val: int):
        self._clip_limit = float(val) / 100.0

    def _tile_size_callback(self, val: int):
        self._tile_size = (val + 1, val + 1)

    def __call__(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return preprocessing.equalize_luminance(img, self._clip_limit, self._tile_size) if self._enable else img


class ContrastGUI:
    def __init__(self):
        self._clip_limit = 1.0
        self._tile_size = (5, 5)
        self._enable = False
        cv2.createTrackbar("Toggle Contrast", "display", 0, 1, self._toggle)
        cv2.createTrackbar("clip", "display", 100, 1000, self._clip_limit_callback)
        cv2.createTrackbar("tile", "display", 5, 15, self._tile_size_callback)

    def _toggle(self, val):
        self._enable = val

    def _clip_limit_callback(self, val: int):
        self._clip_limit = float(val) / 100.0

    def _tile_size_callback(self, val: int):
        self._tile_size = (val + 1, val + 1)

    def __call__(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return preprocessing.equalize_color(img, self._clip_limit, self._tile_size) if self._enable else img


class RebalanceGUI:
    def __init__(self):
        self._gain = [1.0, 1.0, 1.0]
        self._interp = interp1d([0.0, 1000.0], [0.0, 1.0])
        self._enable = False
        cv2.createTrackbar("Toggle rebalance", "display", 0, 1, self._toggle)
        cv2.createTrackbar("Gain R", "display", 1000, 1000, self._gain_r_callback)
        cv2.createTrackbar("Gain G", "display", 1000, 1000, self._gain_g_callback)
        cv2.createTrackbar("Gain B", "display", 1000, 1000, self._gain_b_callback)

    def _toggle(self, val):
        self._enable = val

    def _gain_r_callback(self, val: int):
        self._gain[2] = float(self._interp(float(val)))

    def _gain_g_callback(self, val: int):
        self._gain[1] = float(self._interp(float(val)))

    def _gain_b_callback(self, val: int):
        self._gain[0] = float(self._interp(float(val)))

    def __call__(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return preprocessing.rebalance_color(img, self._gain[2], self._gain[1], self._gain[0]) if self._enable else img


class ImadjustGUI:
    def __init__(self):
        self._gamma = [1.0, 1.0, 1.0]
        self._percent = [100.0, 100.0, 100.0]
        self._interp = interp1d([0.0, 1000.0], [0.0, 10.0])
        self._enable = False
        cv2.createTrackbar("Toggle imadjust", "display", 0, 1, self._toggle)
        cv2.createTrackbar("Gamma R", "display", 100, 1000, self._gamma_r_callback)
        cv2.createTrackbar("Gamma G", "display", 100, 1000, self._gamma_g_callback)
        cv2.createTrackbar("Gamma B", "display", 100, 1000, self._gamma_b_callback)
        cv2.createTrackbar("% R", "display", 1000, 1000, self._percent_r_callback)
        cv2.createTrackbar("% G", "display", 1000, 1000, self._percent_g_callback)
        cv2.createTrackbar("% B", "display", 1000, 1000, self._percent_b_callback)

    def _toggle(self, val):
        self._enable = val

    def _gamma_r_callback(self, val: int):
        self._gamma[2] = float(self._interp(float(val)))

    def _gamma_g_callback(self, val: int):
        self._gamma[1] = float(self._interp(float(val)))

    def _gamma_b_callback(self, val: int):
        self._gamma[0] = float(self._interp(float(val)))

    def _percent_r_callback(self, val: int):
        self._percent[2] = 100.0 * (float(val)) / 1000

    def _percent_g_callback(self, val: int):
        self._percent[1] = 100.0 * (float(val)) / 1000

    def _percent_b_callback(self, val: int):
        self._percent[0] = 100.0 * (float(val)) / 1000

    def __call__(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return preprocessing.imadjust(img, self._percent, self._gamma) if self._enable else img


class UnsharpenGUI:
    def __init__(self):
        self._enable = False
        self._kernel_size = (3, 3)
        self._sigma = 1
        cv2.createTrackbar("Toggle unsharpen", "display", 0, 1, self._toggle)
        cv2.createTrackbar("Kernel Size", "display", 3, 15, self._kernel_size_callback)
        cv2.createTrackbar("Sigma", "display", 1, 20, self._sigma_callback)

    def _toggle(self, val):
        self._enable = val

    def _kernel_size_callback(self, val: int):
        val = max(1, val)
        if val % 2 == 0:
            val = val + 1
        self._kernel_size = (val, val)

    def _sigma_callback(self, val: int):
        self._sigma = val

    def __call__(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return preprocessing.unsharpen_image(img, self._kernel_size, self._sigma) if self._enable else img


class ProcessingViewer:
    def __init__(self, img: npt.NDArray[np.uint8]):
        self._img = img.copy()
        self._h, self._w = self._img.shape[:2]
        cv2.namedWindow("display", cv2.WINDOW_NORMAL)
        cv2.setWindowTitle("display", "Preprocessing Viewer")
        self._processing_steps = (RebalanceGUI(), ContrastGUI(), HSVGUI(), LuminanceGUI(), ImadjustGUI(), UnsharpenGUI())

    def play(self):
        k = -1
        while k != 27 and cv2.getWindowProperty('display', cv2.WND_PROP_VISIBLE) >= 1:
            img = self._img.copy()
            for obj in self._processing_steps:
                img = obj(img)

            # Calculate histograms
            color = ('b', 'g', 'r')
            hist_height = 256
            hist_image = np.zeros((hist_height, img.shape[1], 3), dtype=np.uint8)

            for i, col in enumerate(color):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
                hist = np.int32(np.around(hist))
                pts = np.column_stack((np.arange(hist_height), hist))
                cv2.polylines(hist_image, [pts], False, (i * 100, 128, 255 - i * 100))

            # Flip the histogram image vertically
            hist_image = np.flipud(hist_image)

            # Convert histograms to BGR
            hist_image = cv2.cvtColor(hist_image, cv2.COLOR_RGB2BGR)
            hist_image = np.concatenate((np.zeros((img.shape[0]-hist_image.shape[0], img.shape[1], 3), np.uint8), hist_image), axis=0)

            # Combine the image with the histograms
            img[np.where(hist_image)] = hist_image[np.where(hist_image)]

            # Show the image
            cv2.imshow("display", np.concatenate((self._img, img), axis=1))
            k = cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser(description='Demonstrate the effects of image preprocessing functions.')
    parser.add_argument('input_image', type=Path, help='Path to the input image.')
    args = parser.parse_args()

    args.input_image.resolve(True)

    # Load the input image
    img = cv2.imread(str(args.input_image))

    viewer = ProcessingViewer(img)
    viewer.play()


if __name__ == '__main__':
    main()
