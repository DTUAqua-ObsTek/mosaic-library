import cv2
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from mosaicking.preprocessing import *
from mosaicking.utils import *
from mosaicking.transformations import *
from mosaicking.registration import *


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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--calibration", type=str, default=None, help="Path to calibration file.")
    group.add_argument("-k", "--intrinsic", nargs=9, type=float, default=None, help="Space delimited list of intrinsic matrix terms, Read as K[0,0],K[1,0],K[2,0],K[1,0],K[1,1],K[1,2],K[2,0],K[2,1],K[2,2]")
    parser.add_argument("-d", "--distortion", nargs=4, type=float, default=None, help="Space delimited list of distortion coefficients, Read as K1, K2, p1, p2")
    parser.add_argument("-x", "--xrotation", type=float, default=0, help="Rotation around image plane's x axis (radians).")
    parser.add_argument("-y", "--yrotation", type=float, default=0, help="Rotation around image plane's y axis (radians).")
    parser.add_argument("-z", "--zrotation", type=float, default=0, help="Rotation around image plane's z axis (radians).")
    parser.add_argument("-g", "--gradientclip", type=float, default=0, help="Clip the gradient of severely distorted image.")
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

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Camera Intrinsic Matrix
    if args.intrinsic is not None:
        K = np.array(args.intrinsic).reshape((3,3)).T
    elif args.calibration is not None:
        K = np.eye(3)
        K[0,2] = float(width)/2
        K[1,2] = float(height)/2
    else:
        K = np.eye(3)
        K[0, 2] = float(width) / 2
        K[1, 2] = float(height) / 2
    print("K: {}".format(repr(K)))

    # Camera Lens distortion coefficients
    distCoeff = np.zeros((4, 1), np.float64)
    if args.distortion is not None:
        distCoeff[0, 0] = args.distortion[0]
        distCoeff[1, 0] = args.distortion[1]
        distCoeff[2, 0] = args.distortion[2]
        distCoeff[3, 0] = args.distortion[3]

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
            img = cv2.undistort(img, K, distCoeff)
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
                    R = Rotation.from_euler("xyz",R)
                    img, image_mask, kp_prev = apply_rotation(img, K, R.as_matrix(), kp_prev, gradient_clip=args.gradientclip)
                    if args.show_rotation:
                        cv2.imshow("ROTATION COMPENSATION", img)
                    mosaic_mask = image_mask.copy()
                else:
                    R = Rotation.from_euler("xyz", [args.xrotation, args.yrotation, args.zrotation])
                    img, image_mask, kp_prev = apply_rotation(img, K, R.as_matrix(), kp_prev, gradient_clip=args.gradientclip)
                    mosaic_mask = image_mask.copy()
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
                    img, image_mask, kp = apply_rotation(img, K, R.as_matrix(), kp, gradient_clip=args.gradientclip)
                    if args.show_rotation:
                        cv2.imshow("ROTATION COMPENSATION", img)
                else:
                    R = Rotation.from_euler("xyz", [args.xrotation, args.yrotation, args.zrotation])
                    img, image_mask, kp = apply_rotation(img, K, R.as_matrix(), kp,gradient_clip=args.gradientclip)
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

            # Warp the previous image keypoints into the mosaic's plane
            if A is not None:
                dst_pts = cv2.perspectiveTransform(dst_pts, np.concatenate((A,np.array([[0, 0, 1]]))))

            # Update the homography from current image to mosaic
            A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

            # Get the corners of the current image in homogeneous coords (x,y,w=1)
            src_crn = np.array([[0, img.shape[1], img.shape[1], 0],
                                [0, 0, img.shape[0], img.shape[0]],
                                [1, 1, 1, 1]], float)

            warp_dst = A @ src_crn
            # # Warp the corners
            # xgrid = np.arange(0, width - 1)
            # ygrid = np.arange(0, height - 1)
            # xx, yy = np.meshgrid(xgrid, ygrid, indexing='ij')
            # grid = np.stack((xx.flatten(), yy.flatten(), np.ones_like(yy.flatten())), 0)
            # warp_dst = A @ grid
            # if args.gradientclip > 0:
            #     grad = np.gradient(warp_dst, axis=1)
            #     idx = np.sqrt((grad ** 2).sum(axis=0)) < args.gradientclip
            #     warp_dst = warp_dst[:, idx]

            # Get the corners of the mosaic image in homogeneous coords (x,y,w=1)
            dst_crn = np.array([[0, mosaic_img.shape[1], mosaic_img.shape[1], 0],
                                [0, 0, mosaic_img.shape[0], mosaic_img.shape[0]],
                                [1, 1, 1, 1]], float)

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