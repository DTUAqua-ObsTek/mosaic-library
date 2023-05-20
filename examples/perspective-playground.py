import cv2
import numpy as np
from numpy.typing import NDArray
import argparse

"""Obtained from discussion here: https://stackoverflow.com/questions/45811421/python-create-image-with-new-camera-position"""

class PerspectiveViewer:
    def __init__(self, img: NDArray[np.uint8]):
        self._img = img.copy()
        self._h, self._w = self._img.shape[:2]
        cv2.namedWindow("display", cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowTitle("display", "Perspective Viewer")
        self._f = 500
        self._rot_xyz = np.array([180, 180, 180], float)
        self._dist_xyz = np.array([self._w * 2, self._h * 2, 500], float)

        cv2.createTrackbar("f", "display", self._f, 1000, self._f_callback)
        cv2.createTrackbar("Rotation X", "display", int(self._rot_xyz[0]), 360, self._rot_x_callback)
        cv2.createTrackbar("Rotation Y", "display", int(self._rot_xyz[1]), 360, self._rot_y_callback)
        cv2.createTrackbar("Rotation Z", "display", int(self._rot_xyz[2]), 360, self._rot_z_callback)
        cv2.createTrackbar("Distance X", "display", int(self._dist_xyz[0]), self._w * 4, self._dist_x_callback)
        cv2.createTrackbar("Distance Y", "display", int(self._dist_xyz[1]), self._h * 4, self._dist_y_callback)
        cv2.createTrackbar("Distance Z", "display", int(self._dist_xyz[2]), 1000, self._dist_z_callback)

    def _f_callback(self, val):
        self._f = val

    def _rot_x_callback(self, val):
        self._rot_xyz[0] = val - 180

    def _rot_y_callback(self, val):
        self._rot_xyz[1] = val - 180

    def _rot_z_callback(self, val):
        self._rot_xyz[2] = val - 180

    def _dist_x_callback(self, val):
        self._dist_xyz[0] = val

    def _dist_y_callback(self, val):
        self._dist_xyz[1] = val

    def _dist_z_callback(self, val):
        self._dist_xyz[2] = val

    def play(self):
        k = -1
        while k != 27 and cv2.getWindowProperty('display', cv2.WND_PROP_VISIBLE) >= 1:
            if self._f <= 0:
                self._f = 1
            rotX = (self._rot_xyz[0]) * np.pi / 180  # convert to radians
            rotY = (self._rot_xyz[1]) * np.pi / 180
            rotZ = (self._rot_xyz[2]) * np.pi / 180
            distX = self._dist_xyz[0] - self._w * 2
            distY = self._dist_xyz[1] - self._h * 2
            distZ = self._dist_xyz[2] - 500

            # Camera intrinsic matrix (2D homogeneous matrix)
            K = np.array([[self._f, 0, self._w / 2, 0],
                          [0, self._f, self._h / 2, 0],
                          [0, 0, 1, 0]])

            # K inverse, here we are converting a 2D point on the image place into a 3D ray (4x1 vector), therefore
            # K^-1 must be a 4x3 matrix (4x3 @ 3x1 = 4x1)
            Kinv = np.zeros((4, 3))
            Kinv[:3, :3] = np.linalg.inv(K[:3, :3]) * self._f
            Kinv[-1, :] = [0, 0, 1]

            # Rotation matrices around the X,Y,Z axis
            RX = np.array([[1, 0, 0, 0],
                           [0, np.cos(rotX), -np.sin(rotX), 0],
                           [0, np.sin(rotX), np.cos(rotX), 0],
                           [0, 0, 0, 1]])

            RY = np.array([[np.cos(rotY), 0, np.sin(rotY), 0],
                           [0, 1, 0, 0],
                           [-np.sin(rotY), 0, np.cos(rotY), 0],
                           [0, 0, 0, 1]])

            RZ = np.array([[np.cos(rotZ), -np.sin(rotZ), 0, 0],
                           [np.sin(rotZ), np.cos(rotZ), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

            # Composed rotation matrix with (RX,RY,RZ)
            R = np.linalg.multi_dot([RX, RY, RZ])

            # Translation matrix (homogeneous) (pixels)
            T = np.array([[distX], [distY], [distZ], [1.0]])

            RT = np.c_[R[:, :3], R@T]

            H = K.dot(RT.dot(Kinv))

            # Apply matrix transformation
            dst = cv2.warpPerspective(src, H, (self._w, self._h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,0,0))

            # Show the image
            cv2.imshow("display", dst)
            k = cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Example Perspective Viewer")
    parser.add_argument("filepath", type=str, help="Path to the image file to view.")
    args = parser.parse_args()

    #Read input image, and create output image
    src = cv2.imread(str(args.filepath))
    #src = cv2.resize(src, (640,480))
    viewer = PerspectiveViewer(src)
    try:
        viewer.play()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
