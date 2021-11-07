import keyboard
import numpy as np
from mosaicking.transformations import ImageTransformer, bird_view
import cv2
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str, help="Path to the image file that needs correcting.")
args = parser.parse_args()


def apply_rotation(img: np.ndarray, rotation: np.ndarray):
    """Apply a 3D rotation to an image and keypoints, treating them as if on a plane."""
    H, bounds = get_rotation_homography(img, rotation.T)
    out = cv2.warpPerspective(img, H, bounds)
    return out


def get_rotation_homography(img: np.ndarray, rotation: np.ndarray):
    """Given a rotation, obtain the homography and the new bounds of the rotated image."""
    # Acquire the four corners of the image
    X = np.array([[0, 0, img.shape[1] / 2],
                  [img.shape[1], 0, img.shape[1] / 2],
                  [img.shape[1], img.shape[0], img.shape[1] / 2],
                  [0, img.shape[0], img.shape[1] / 2]], dtype="float32").T
    X1 = rotation @ X  # Rotate the coordinates
    H, _ = cv2.findHomography(X[:2, :].T, X1[:2, :].T, cv2.RANSAC)  # Calculate the homography of the transformation
    # Calculate the new bounds
    xmin, ymin, _ = np.int32(X1.min(axis=1) - 0.5)
    xmax, ymax, _ = np.int32(X1.max(axis=1) + 0.5)
    # Apply a C homography
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # C homography
    return Ht.dot(H), (xmax - xmin, ymax - ymin)


img = cv2.imread(str(Path(args.filepath).resolve()))
width,height,channels = img.shape
tformer = ImageTransformer(img)
warped = img.copy()
cv2.namedWindow("W/S to tilt around X axis. A/D to tilt around Y axis. Q/E to tilt around Z axis.", cv2.WINDOW_AUTOSIZE)
cv2.imshow("W/S to tilt around X axis. A/D to tilt around Y axis. Q/E to tilt around Z axis.", warped)
cv2.waitKey(1)
euler = np.zeros((3,),dtype="float32")
euler[0] = 45
scaler = 1.0
change = False
while True:
    key = keyboard.read_event()
    if key.name+key.event_type == "wdown":
        change = True
        euler[1]+=scaler
    elif key.name+key.event_type == "sdown":
        change = True
        euler[1]-=scaler
    elif key.name + key.event_type == "adown":
        change = True
        euler[2]+=scaler
    elif key.name + key.event_type == "ddown":
        change = True
        euler[2]-=scaler
    elif key.name + key.event_type == "qdown":
        change = True
        euler[0]+=scaler
    elif key.name + key.event_type == "edown":
        change = True
        euler[0]-=scaler
    elif key.name + key.event_type == "page updown":
        change = True
        scaler+=1
    elif key.name + key.event_type == "pgdndown":
        change = True
        scaler-=1
    elif key.name + key.event_type == "escdown":
        break
    elif key.name + key.event_type == "spacedown":
        change = True
        euler = np.zeros((3,), dtype="float32")
        scaler = 1.0
    else:
        print(key.name + key.event_type)
    if change:
        r = Rotation.from_euler('xyz', euler.tolist(), degrees=True)
        R = r.as_matrix()
        warped = apply_rotation(img, R)
        #warped = bird_view(img, int(euler[1]))
        change = False
    cv2.imshow("W/S to tilt around X axis. A/D to tilt around Y axis. Q/E to tilt around Z axis.", warped)
    cv2.waitKey(1)
cv2.destroyAllWindows()
