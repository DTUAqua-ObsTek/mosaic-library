import numpy as np
import cv2
import argparse
from pathlib import Path
import json
import sys


parser = argparse.ArgumentParser()
parser.add_argument("video", type=str, help="Path to video file.")
parser.add_argument("calibration", type=str, help="Path to output json file containing calibration information.")
args = parser.parse_args()

video_path = Path(args.video).resolve()
assert video_path.exists(), "File not found: {}".format(str(video_path))

reader = cv2.VideoCapture(str(video_path))
fps = reader.get(cv2.CAP_PROP_FPS)
width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

try:
    for frame in range(nframes):
        ret, img = reader.read()
        if not ret:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)
except Exception as err:
    print(repr(err), file=sys.stderr)
finally:
    reader.release()
    cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
output = {"K": mtx.flatten(),
          "D": dist.flatten()}
out = Path(args.calibration).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
with open(args.calibration, "w") as f:
    json.dump(output, f)