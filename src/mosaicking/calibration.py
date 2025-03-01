import numpy as np
import cv2
import argparse
from pathlib import Path
import json
from sklearn.cluster import KMeans
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(filename)s:%(message)s")
    model_flags = {"radtan": None, "ratpoly": cv2.CALIB_RATIONAL_MODEL}
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to video file.")
    parser.add_argument("calibration", type=str, help="Path to output json file containing calibration information.")
    parser.add_argument("--cb_pattern", type=int, nargs=2, default=(8, 6), help="Number of vertices (cols rows)")
    parser.add_argument("--square_size", type=float, default=30.0, help="Size of the squares in mm.")
    parser.add_argument("--reduction_fraction", type=float, default=None, help="Portion of samples to keep.")
    parser.add_argument("--model", type=str, default="radtan", choices=model_flags.keys(),
                        help="Choose either radial-tangent or rational polynomial models.")
    args = parser.parse_args()
    video_path = Path(args.video).resolve(True)
    if args.reduction_fraction is not None:
        assert 0.0 < args.reduction_fraction < 1.0, "Reduction fraction must be in interval 0 < reduction_fraction < 1.0"

    reader = cv2.VideoCapture(str(video_path))
    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    cols, rows = args.cb_pattern
    square_size_mm = args.square_size

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[:rows, :cols].T.reshape(-1, 2)
    objp = objp * square_size_mm
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = np.full((nframes, cols * rows, 1, 2), np.nan, np.float32)  # 2d points in image plane.
    try:
        logger.info("Step 1: Extracting Corners.")
        for frame in range(nframes):
            ret, img = reader.read()  # get the next frame
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (rows, cols), cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS)
            # If found, add object points, image points (after refining them)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints[frame, ...] = corners2
            progress = frame / (nframes - 1)
            logger.info(f"Corner Extraction: [{'=' * int(progress * 50):<50}] {int(progress * 100)}%")
                # Draw and display the corners
                #cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(1)
    except Exception as err:
        logger.error(repr(err))
        raise err
    finally:
        reader.release()
        #cv2.destroyAllWindows()
    logger.info(f"Corner Extraction: [{'=' * int(1.0 * 50):<50}] {int(1.0 * 100)}% Done!",)
    imgpoints = imgpoints[~np.isnan(imgpoints).reshape((nframes, cols*rows*2)).any(axis=1)]
    nsamples = imgpoints.shape[0]
    if args.reduction_fraction is not None and args.reduction_fraction < 1:
        logger.info("Step 2: Clustering corners to reduce samples.")
        nsamples_reduced = int(args.reduction_fraction * nsamples)
        kmeans = KMeans(nsamples_reduced, n_init='auto')
        kmeans.fit(imgpoints.reshape((nsamples, cols*rows*2)))
        labels = kmeans.predict(imgpoints.reshape((nsamples, cols*rows*2)))
        imgpoints_reduced = np.full((nsamples_reduced, cols*rows, 1, 2), np.nan, np.float32)
        for sample in range(nsamples_reduced):
            cluster_idx = np.argwhere(labels.squeeze() == sample).squeeze()
            imgpoints_reduced[sample, ...] = imgpoints[np.random.choice(cluster_idx, 1), ...]
    else:
        logger.info("Step 2: Skipping because reduction_fraction is not set.")
        nsamples_reduced = nsamples
        imgpoints_reduced = imgpoints.copy()
    logger.info("Step 3: Calibrating.")
    objpoints = [objp for _ in range(nsamples_reduced)]
    imgpoints_reduced = [p.squeeze() for p in imgpoints_reduced]
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_reduced, (width, height), None, None)
    logger.info(f"Results\nRMS re-projection error: {ret} pix.")
    logger.info(f"Obtained Intrinsic Matrix\n{K}\nObtained Distortion Coefficients\n{D}")
    output = {"K": K.flatten().tolist(),
              "D": D.flatten().tolist()}
    out = Path(args.calibration).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {out}")
    with open(args.calibration, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
