import cv2
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from mosaicking.preprocessing import *


def load_orientations(path: os.PathLike, args):
    """Given a path containing orientations, retrieve the orientations corresponding to a time offset between video and orientation data."""
    time_offset = args.time_offset if args.time_offset else args.sync_points[1] - args.sync_points[0]
    df = pd.read_csv(str(path), index_col="timestamp")
    df.index = df.index - time_offset
    return df[~df.duplicated()]


def get_features(img: np.ndarray, fdet: cv2.Feature2D, mask=None):
    """Given a feature detector, obtain the features found in the image."""
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return fdet.detectAndCompute(img, mask)


def get_starting_pos(cap: cv2.VideoCapture, args):
    """Set a VideoCapture object to a position (either in seconds or the frame #)"""
    if args.start_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000.0)
    elif args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame - 1)
    return cap


def evaluate_stopping(cap: cv2.VideoCapture, args):
    """Return true if stopping conditions met."""
    if args.finish_time:
        return cap.get(cv2.CAP_PROP_POS_MSEC) > args.finish_time*1000.0
    elif args.finish_frame:
        return cap.get(cv2.CAP_PROP_POS_FRAMES) > args.finish_frame-1
    return cap.get(cv2.CAP_PROP_FRAME_COUNT)-1 <= cap.get(cv2.CAP_PROP_POS_FRAMES)


def apply_rotation(img: np.ndarray, rotation: np.ndarray, keypoints: list, scale_factor: float = None):
    """Apply a 3D rotation to an image and keypoints, treating them as if on a plane."""
    H, bounds = get_rotation_homography(img, rotation.T)
    out = cv2.warpPerspective(img, H, bounds)
    mask = cv2.warpPerspective(np.ones(img.shape[:2], dtype='uint8'), H, bounds)
    pts = [k.pt for k in keypoints]
    pts = H @ np.concatenate((np.array(pts), np.ones((len(pts), 1))), axis=1).T
    pts = pts[:2, :].T
    for k, pt in zip(keypoints, pts):
        k.pt = tuple(pt)
    return out, mask, keypoints


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
    # Apply a translation homography
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translation homography
    return Ht.dot(H), (xmax - xmin, ymax - ymin)


def scale_img(img: np.ndarray, keypoints: list, scale: float):
    if scale < 1:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    S = np.eye(3, dtype=float)
    S[0,0] = scale
    S[1,1] = scale
    pts = [k.pt for k in keypoints]
    pts = S @ np.concatenate((np.array(pts), np.ones((len(pts), 1))), axis=1).T
    pts = pts[:2, :].T
    for k, pt in zip(keypoints, pts):
        k.pt = tuple(pt)
    return img, keypoints


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to video file.")
    parser.add_argument("--output_directory", type=str, help="Path to directory where output mosaics are to be saved. Default is invokation path.", default=".")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start_time", type=float, help="Time (secs) to start from.")
    group.add_argument("--start_frame", type=int, help="Frame number to start from.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--finish_time", type=float, help="Time (secs) to finish at.")
    group.add_argument("--finish_frame", type=int, help="Frame number to finish at.")
    parser.add_argument("--frame_skip", type=int, default=None, help="Number of frames to skip between each mosaic update.")
    parser.add_argument("--orientation_file", type=str, default=None, help="Path to .csv file containing orientation measurements.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sync_points", type=float, nargs=2, default=None, help="Time points (sec) where video and orientation file are in sync, used to calculate time offset between video and timestamps in orientation file.")
    group.add_argument("--time_offset", type=float, default=None, help="Time offset (sec) between video and orientation file timestamps, used for synchronization.")
    parser.add_argument("--min_matches", type=int, default=4, help="Minimum number of matches to proceed with registration.")
    parser.add_argument("--min_features", type=int, default=100, help="Minimum number of features to detect in an image.")
    parser.add_argument("--max_warp_size", type=int, nargs='+', default=None, help="Maximum size of warped image (used to prevent OOM errors), if 1 argument given then image is clipped to square, if 2 then the order is height, width.")
    parser.add_argument("--max_mosaic_size", type=int, default=None, help="Largest allowable size (width or height) for mosaic. Creates a new tile if it gets bigger.")
    parser.add_argument("--save_freq", type=int, default=0, help="Save frequency for output mosaic (if less than 1 then output saves at exit).")
    parser.add_argument("--scale_factor", type=float, default=0.0, help="Scale the input image with constant aspect ratio.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha blending scalar for merging new frames into mosaic.")
    parser.add_argument("--show_rotation", action="store_true", help="Flag to display the rotation compensation using rotation data.")
    parser.add_argument("--show_mosaic", action="store_true", help="Flag to display the mosaic output.")
    parser.add_argument("--show_preprocessing", action="store_true", help="Flag to display the preprocessed image")
    parser.add_argument("--fix_color", action="store_true", help="Flag to preprocess image for color balance.")
    parser.add_argument("--fix_contrast", action="store_true", help="Flag to preprocess image for contrast equalization.")
    parser.add_argument("--fix_light", action="store_true", help="Flag to preprocess image for lighting equalization.")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    assert video_path.exists(), "File not found: {}".format(str(video_path))
    output_path = Path(args.output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if args.orientation_file is not None:
        ori_path = Path(args.orientation_file).resolve()
        assert ori_path.exists(), "File not found: {}".format(str(ori_path))
        if args.sync_points is None and args.time_offset is None:
            sys.stderr.write("Warning: No --sync_points or --time_offset argument given, assuming video and orientation file start at the same time.\n")
        orientations = load_orientations(ori_path, args)
        quat_lut = interp1d(orientations.index, orientations[["qx","qy","qz","qw"]],axis=0)

    reader = cv2.VideoCapture(str(video_path))
    reader = get_starting_pos(reader, args)
    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    formatspec = "{:0"+"{}d".format(len(str(n_frames)))+"}"
    if args.show_mosaic:
        cv2.namedWindow(str(video_path), cv2.WINDOW_NORMAL)
    if args.show_rotation:
        cv2.namedWindow("ROTATION COMPENSATION", cv2.WINDOW_NORMAL)
    if args.show_preprocessing:
        cv2.namedWindow("PREPROCESSING RESULT", cv2.WINDOW_NORMAL)

    detector = cv2.ORB_create(nfeatures=500)  # ORB detector is pretty good and is CC licensed
    # detector = cv2.SIFT_create(nfeatures=500)  # SIFT detector performs better but is patented by David Lowe

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Camera Lens distortion coefficients (guesstimated)
    distCoeff = np.zeros((4, 1), np.float64)
    k1 = -1.0e-5  # negative to remove barrel distortion
    k2 = 0
    p1 = 0.0
    p2 = 0.0
    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2

    # Camera Intrinsic Matrix (also guesstimated)
    # assume unit matrix for camera
    cam = np.eye(3, dtype=np.float32)
    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2.0  # define center y
    cam[0, 0] = 10  #715.2699  # define focal length x
    cam[1, 1] = 10  #711.5281  # define focal length y
    # cam[2, 0] = 575.6995
    # cam[2, 1] = 366.3466


    # BEGIN MAIN LOOP #
    first = True
    counter = 0
    tile_counter = 0
    try:  # External try to handle any unexpected errors
        while not evaluate_stopping(reader, args):
            sys.stdout.write("Processing Frame {}\n".format(int(reader.get(cv2.CAP_PROP_POS_FRAMES)+1)))
            if args.frame_skip is not None and not first:
                reader.set(cv2.CAP_PROP_POS_FRAMES, reader.get(cv2.CAP_PROP_POS_FRAMES)+args.frame_skip-1)
            # Acquire a frame
            ret, img = reader.read()
            og = img.copy()  # keep a copy for later reference

            if not ret:
                sys.stderr.write("Frame missing: {}\n".format(formatspec.format(int(reader.get(cv2.CAP_PROP_POS_FRAMES)))))
                continue

            # Preprocess the image
            img = cv2.undistort(img, cam, distCoeff)
            img = fix_color(img) if args.fix_color else img
            img = fix_contrast(img) if args.fix_contrast else img
            img = fix_light(img) if args.fix_light else img
            if args.show_preprocessing:
                cv2.imshow("PREPROCESSING RESULT", np.concatenate((og, img)))

            if first:
                # Detect keypoints on the first frame
                kp_prev, des_prev = get_features(img, detector)
                if len(kp_prev) < args.min_features:
                    sys.stderr.write("Not Enough Features, Finding Good Frame.\n")
                    continue
                # Apply scaling to image if specified
                if args.scale_factor > 0.0:
                    img, kp_prev = scale_img(img, kp_prev, args.scale_factor)
                # Apply rotation to image if necessary
                if args.orientation_file is not None:
                    lookup_time = reader.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    R = Rotation.from_quat(quat_lut(lookup_time))
                    R = R.as_euler("xyz")
                    R = R[[0,1,2]] * np.array([1, 1, 1]) + np.array([0, 0, 0])
                    R = Rotation.from_euler("xyz",R)
                    img, image_mask, kp_prev = apply_rotation(img, R.as_matrix(), kp_prev)
                    if args.show_rotation:
                        cv2.imshow("ROTATION COMPENSATION", img)
                    mosaic_mask = image_mask.copy()
                else:
                    image_mask = np.ones(img.shape[:2], np.uint8)
                    mosaic_mask = np.ones(img.shape[:2], np.uint8)
                mosaic_img = img.copy()  # initialize the mosaic
                prev_img = img.copy()  # store the image as previous
                first = False
                A = None
                continue
            else:
                # Detect keypoints in the new frame
                kp, des = get_features(img, detector)
                if len(kp) < args.min_features:
                    sys.stderr.write("Not Enough Features, Skipping Frame.\n")
                    first = True
                    sys.stdout.write("Cropping tile.\n")
                    fpath = output_path.joinpath("tile_{:03d}.png".format(tile_counter))
                    cv2.imwrite(str(fpath), mosaic_img)
                    tile_counter = tile_counter + 1
                    continue
                if args.scale_factor > 0.0:
                    img, kp = scale_img(img, kp, args.scale_factor)
                if args.orientation_file is not None:
                    lookup_time = reader.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    R = Rotation.from_quat(quat_lut(lookup_time))
                    R = R.as_euler("xyz")
                    R = R[[0,1,2]] * np.array([1, 1, 1]) + np.array([0, 0, 0])
                    R = Rotation.from_euler("xyz", R)
                    img, image_mask, kp = apply_rotation(img, R.as_matrix(), kp)
                    if args.show_rotation:
                        cv2.imshow("ROTATION COMPENSATION", img)

            if len(kp) < args.min_features:
                sys.stderr.write("Not Enough Features, Skipping Frame.\n")
                continue

            # Match descriptors between previous and new
            knn_matches = flann.knnMatch(des_prev.astype(np.float32), des.astype(np.float32), 2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in knn_matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) < args.min_matches:
                sys.stderr.write("Not Enough Matches, Skipping Frame.\n")
                continue

            # Current Image Keypoints
            src_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            # Previous Image Keypoints
            dst_pts = np.float32([ kp_prev[m.queryIdx].pt for m in good ]).reshape(-1,1,2)

            # Warp the destination keypoints into the mosaic computing the Similarity tranform
            if A is not None:
                dst_pts = cv2.perspectiveTransform(dst_pts, np.concatenate((A,np.array([[0, 0, 1]]))))

            # Update the homography from current image to mosaic
            # A, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # A, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
            A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
            matchesMask = mask.ravel().tolist()

            # Get the corners of the current image in homogeneous coords (x,y,w=1)
            src_crn = np.array([[0, img.shape[1], img.shape[1], 0],
                                [0, 0, img.shape[0], img.shape[0]],
                                [1, 1, 1, 1]], float)
            # Get the corners of the mosaic image in homogeneous coords (x,y,w=1)
            dst_crn = np.array([[0, mosaic_img.shape[1], mosaic_img.shape[1], 0],
                                [0, 0, mosaic_img.shape[0], mosaic_img.shape[0]],
                                [1, 1, 1, 1]], float)

            warp_dst = A @ src_crn

            # Concatenate the mosaic and warped corner coordinates
            pts = np.concatenate([dst_crn[:2,:], warp_dst], axis=1)

            # Round to pixel centers
            xmin,ymin = np.int32(pts.min(axis=1) - 0.5)
            xmax,ymax = np.int32(pts.max(axis=1) + 0.5)

            t = [-xmin, -ymin]  # calculate translation
            A[:,-1] = A[:,-1]+t  # translation homography

            if args.max_mosaic_size is not None:
                if xmax-xmin > args.max_mosaic_size or ymax-ymin > args.max_mosaic_size:
                    sys.stdout.write("Cropping tile.\n")
                    first = True
                    fpath = output_path.joinpath("tile_{:03d}.png".format(tile_counter))
                    cv2.imwrite(str(fpath), mosaic_img)
                    tile_counter = tile_counter + 1
                    continue

            # warp the input image into the mosaic's plane
            warped = cv2.warpAffine(img, A, (xmax - xmin, ymax - ymin))
            warped_mask = cv2.warpAffine(image_mask, A, (xmax - xmin, ymax - ymin))

            # Get the previous iteration mosaic_mask in the shape of the update
            template = cv2.warpAffine(np.zeros(img.shape, np.uint8), A, (xmax - xmin, ymax - ymin))
            mosaic_mask_ = template[:, :, 0].copy()
            mosaic_mask_[t[1]:mosaic_mask.shape[0] + t[1], t[0]:mosaic_mask.shape[1] + t[0]] = mosaic_mask
            mosaic_img_ = template.copy()
            mosaic_img_[t[1]:mosaic_img.shape[0] + t[1], t[0]:mosaic_img.shape[1] + t[0]] = mosaic_img

            # mosaic only
            mosaic_only = cv2.bitwise_and(mosaic_mask_, cv2.bitwise_not(warped_mask))
            # intersection
            shared = cv2.bitwise_and(warped_mask, mosaic_mask_)
            # Combine the image and tile, and update
            mosaic_img = np.where(cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR) > 0, warped, mosaic_img_)
            mosaic_img = np.where(cv2.cvtColor(mosaic_only, cv2.COLOR_GRAY2BGR) > 0, mosaic_img_, mosaic_img)
            mixer = np.uint8(args.alpha * mosaic_img_.astype(np.float32) + (1.0-args.alpha) * warped.astype(np.float32))
            mosaic_img = np.where(cv2.cvtColor(shared, cv2.COLOR_GRAY2BGR) > 0, mixer, mosaic_img)

            # update the tile_mask
            mosaic_mask = cv2.bitwise_or(mosaic_mask_, warped_mask)

            # Display the mosaic
            if args.show_mosaic:
                cv2.imshow(str(video_path), mosaic_img)

            prev_img = img.copy()  # Update the previous frame
            kp_prev = kp  # Update the previous keypoints
            des_prev = des.copy()  # Update the previous descriptors

            key = cv2.waitKey(10)
            counter = counter + 1
            if args.save_freq > 0 and counter % args.save_freq == 0:
                counter = 0
                cv2.imwrite("output_mosaic.png", mosaic_img)
            if key == ord("q"):
                sys.stdout.write("Quitting.\n")
                break
    except Exception as err:
        # Some strange error has occurred, write out to stderr, cleanup and rethrow the error
        cv2.imwrite("error_img.png", mosaic_img)
        cv2.imwrite("error_frame.png", img)
        sys.stderr.write("\nPipeline failed at frame {}\n".format(reader.get(cv2.CAP_PROP_POS_FRAMES)+1))
        cv2.destroyWindow(str(video_path))
        reader.release()
        raise err
    # The video exited properly, so cleanup and exit.
    fpath = output_path.joinpath("tile_{:03d}.png".format(tile_counter))
    cv2.imwrite(str(fpath), mosaic_img)
    if args.show_mosaic:
        cv2.destroyWindow(str(video_path))
    if args.show_rotation:
        cv2.destroyWindow("ROTATION COMPENSATION")
    reader.release()