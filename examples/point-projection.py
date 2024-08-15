import cv2
from offline import generate_plane, project_world_lines
from utils import parse_args, parse_intrinsics
from scipy.spatial.transform import Rotation
from scipy import interpolate
import numpy as np
import numpy.typing as npt


class GridProjectionViewer:
    def __init__(self, img: npt.NDArray[np.uint8], K: npt.NDArray[float], D: npt.NDArray[float]):
        self._img = img.copy()
        self._h, self._w = self._img.shape[:2]
        self._K = K.copy()
        self._D = D.copy()
        cv2.namedWindow("display", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setWindowTitle("display", "Perspective Viewer")
        self._rot_xyz = np.array([0.0, 0.0, 0.0], float)  # Rotation from Camera to World
        self._pos_xyz = np.array([0.0, 0.0, 0.0], float)  # Translation from Camera to World
        self._rot_interp = interpolate.interp1d([0.0, 3600.0], [-180.0, 180.0])
        self._pos_interp = interpolate.interp1d([0.0, 1000.0], [-5.0, 5.0])
        cv2.createTrackbar("Rotation X", "display", 1800, 3600, self._rot_x_callback)
        cv2.createTrackbar("Rotation Y", "display", 1800, 3600, self._rot_y_callback)
        cv2.createTrackbar("Rotation Z", "display", 1800, 3600, self._rot_z_callback)
        cv2.createTrackbar("Pos X", "display", 500, 1000, self._pos_x_callback)
        cv2.createTrackbar("Pos Y", "display", 500, 1000, self._pos_y_callback)
        cv2.createTrackbar("Pos Z", "display", 500, 1000, self._pos_z_callback)

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
        while k != 27:
            rotX = (self._rot_xyz[0]) * np.pi / 180  # convert to radians
            rotY = (self._rot_xyz[1]) * np.pi / 180
            rotZ = (self._rot_xyz[2]) * np.pi / 180
            distX = self._pos_xyz[0]
            distY = self._pos_xyz[1]
            distZ = self._pos_xyz[2]
            R = Rotation.from_euler("xyz", (rotX, rotY, rotZ))
            T = np.array([distX, distY, distZ])
            grid = generate_plane((-2.0, -2.0, 4, 4), 100, 2)
            projected = project_world_lines(grid, R, self._img.copy(), K, None, T)
            cv2.imshow("display", projected)
            k = cv2.waitKey(1)
        cv2.destroyAllWindows()


img = cv2.imread("/home/fft/Data/AquaLoc/archaeology/frame_00001.png")

grid = generate_plane((-1.0, -1.0, 2, 2), 100)

args = parse_args()
K, D = parse_intrinsics(args, img.shape[1], img.shape[0])

p = GridProjectionViewer(img, K, D)
try:
    p.play()
except KeyboardInterrupt:
    cv2.destroyAllWindows()

