import cv2
import numpy as np
from numpy import typing as npt
from scipy import interpolate
from mosaicking import transformations
import argparse
from typing import Tuple
import sys

"""
Obtained from discussion here: https://stackoverflow.com/questions/45811421/python-create-image-with-new-camera-position
"""


def add_inner_border(image: npt.NDArray[np.uint8], thickness: int, color: Tuple[int, ...]) -> npt.NDArray[np.uint8]:
    # Draw top border
    cv2.rectangle(image, (0, 0), (image.shape[1], thickness), color, thickness)
    # Draw bottom border
    cv2.rectangle(image, (0, image.shape[0]), (image.shape[1], image.shape[0] - thickness), color, thickness)
    # Draw left border
    cv2.rectangle(image, (0, 0), (thickness, image.shape[0]), color, thickness)
    # Draw right border
    cv2.rectangle(image, (image.shape[1], 0), (image.shape[1] - thickness, image.shape[0]), color, thickness)
    return image


class PerspectiveViewer:
    def __init__(self, img: npt.NDArray[np.uint8]):
        self._img = img.copy()
        self._h, self._w = self._img.shape[:2]
        cv2.namedWindow("display", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setWindowTitle("display", "Perspective Viewer")
        self._f = 500
        self._fx = 500
        self._fy = 500
        self._cx = int(self._w / 2)
        self._cy = int(self._h / 2)
        self._rot_xyz = np.array([180, 180, 180], float)
        self._pos_xyz = np.array([0.0, 0.0, 0.5], float)
        self._rot_interp = interpolate.interp1d([0.0, 3600.0], [-180.0, 180.0])
        self._pos_interp = interpolate.interp1d([0.0, 1000.0], [-5.0, 5.0])
        #cv2.createTrackbar("f", "display", self._f, 1000, self._f_callback)
        cv2.createTrackbar("fx", "display", self._fx, 1000, self._fx_callback)
        cv2.createTrackbar("fy", "display", self._fy, 1000, self._fy_callback)
        #cv2.createTrackbar("w", "display", self._cx, self._w, self._cx_callback)
        #cv2.createTrackbar("h", "display", self._cy, self._h, self._cy_callback)
        cv2.createTrackbar("Rotation X", "display", 1800, 3600, self._rot_x_callback)
        cv2.createTrackbar("Rotation Y", "display", 1800, 3600, self._rot_y_callback)
        cv2.createTrackbar("Rotation Z", "display", 1800, 3600, self._rot_z_callback)
        cv2.createTrackbar("Pos X", "display", 500, 1000, self._pos_x_callback)
        cv2.createTrackbar("Pos Y", "display", 500, 1000, self._pos_y_callback)
        cv2.createTrackbar("Pos Z", "display", 501, 1000, self._pos_z_callback)

    def _f_callback(self, val):
        self._f = val

    def _fx_callback(self, val):
        self._fx = val

    def _fy_callback(self, val):
        self._fy = val

    def _cx_callback(self, val):
        self._cx = float(val)

    def _cy_callback(self, val):
        self._cy = float(val)

    def _rot_x_callback(self, val):
        self._rot_xyz[0] = self._rot_interp(float(val))

    def _rot_y_callback(self, val):
        self._rot_xyz[1] = self._rot_interp(float(val))

    def _rot_z_callback(self, val):
        self._rot_xyz[2] = self._rot_interp(float(val))

    def _pos_x_callback(self, val):
        self._pos_xyz[0] = self._pos_interp(float(val))

    def _pos_y_callback(self, val):
        self._pos_xyz[1] = self._pos_interp(float(val))

    def _pos_z_callback(self, val):
        self._pos_xyz[2] = self._pos_interp(float(val))

    def play(self):
        k = -1
        while k != 27 and cv2.getWindowProperty('display', cv2.WND_PROP_VISIBLE) >= 1:
            if self._f <= 0:
                self._f = 1
            rotX = (self._rot_xyz[0]) * np.pi / 180  # convert to radians
            rotY = (self._rot_xyz[1]) * np.pi / 180
            rotZ = (self._rot_xyz[2]) * np.pi / 180
            distX = self._pos_xyz[0]
            distY = self._pos_xyz[1]
            distZ = self._pos_xyz[2]
            # Camera intrinsic matrix (2D homogeneous matrix)
            K = np.array([[500, 0, self._w / 2,],
                          [0, 500, self._h / 2,],
                          [0, 0, 1]])
            Kinv = np.linalg.inv(K)
            Kout = np.array([[self._fx, 0, self._w / 2,],
                             [0, self._fy, self._h / 2,],
                             [0, 0, 1]])
            Rx = transformations.R_x(rotX)
            Ry = transformations.R_y(rotY)
            Rz = transformations.R_z(rotZ)
            R_nadir_cam = Rz @ Ry @ Rx
            T_nadir_cam = np.eye(4)
            E_nadir_cam = T_nadir_cam @ R_nadir_cam
            E_cam_nadir = np.eye(4)
            E_cam_nadir[:3, :3] = E_nadir_cam[:3, :3].T
            E_cam_nadir[:3, -1:] = -E_nadir_cam[:3, :3].T @ E_nadir_cam[:3, -1:]
            # Define nadir to ground and inverse
            R_nadir_gnd = np.eye(4)  # For now, nadir and gnd plane frames are parallel
            T_nadir_gnd = np.eye(4)
            T_nadir_gnd[:3, -1:] = np.array([[distX], [distY], [distZ]])
            E_nadir_gnd = R_nadir_gnd @ T_nadir_gnd
            E_gnd_nadir = np.eye(4)
            E_gnd_nadir[:3, :3] = E_nadir_gnd[:3, :3].T
            E_gnd_nadir[:3, -1:] = -E_nadir_gnd[:3, :3].T @ E_nadir_gnd[:3, -1:]
            E_gnd_cam = E_nadir_cam @ E_gnd_nadir  # Ground to camera = ground->nadir->camera
            E_cam_gnd = E_nadir_gnd @ E_cam_nadir  # Camera to ground = camera->nadir->camera
            crn = np.array([[0.0, self._img.shape[1] - 1, self._img.shape[1] - 1, 0.0],
                            [0.0, 0.0, self._img.shape[0] - 1, self._img.shape[0] - 1]])
            # The destination corners are warped as follows
            dest_crn = Kinv @ cv2.convertPointsToHomogeneous(crn.T[:, None, ...]).squeeze().T
            dest_crn = E_cam_gnd @ cv2.convertPointsToHomogeneous(dest_crn.T).squeeze().T
            dest_crn = Kout @ cv2.convertPointsFromHomogeneous(dest_crn.T).squeeze().T
            dest_crn = cv2.convertPointsFromHomogeneous(dest_crn.T).astype(int)
            # Find the homography from orignal image to new image perspective
            H, _ = cv2.findHomography(crn.T[:, None, ...], dest_crn, method=cv2.RANSAC)
            try:
                warp_img = cv2.warpPerspective(self._img, H, self._img.shape[1::-1], flags=cv2.WARP_INVERSE_MAP)
            except cv2.error as e:
                print("Warning!: Output image is infinitely thin, adjust sliders accordingly!", file=sys.stderr)
                cv2.imshow("display", dst)
                k = cv2.waitKey(1)
                continue
            warp_img = add_inner_border(warp_img, 3, (0, 255, 255))
            og = self._img.copy()
            cv2.polylines(og, [dest_crn], True, (0, 255, 255), 3)
            # Show the images
            dst = np.concatenate((og, warp_img), axis=1)
            cv2.imshow("display", dst)
            k = cv2.waitKey(1)
        cv2.destroyWindow("display")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Example Perspective Viewer")
    parser.add_argument("filepath", type=str, help="Path to the image file to view.")
    args = parser.parse_args()

    #Read input image, and create output image
    src = cv2.imread(str(args.filepath))
    viewer = PerspectiveViewer(src)
    try:
        viewer.play()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
