import cv2
from cv2 import fisheye
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from mosaicking import preprocessing, utils, transformations, registration
from itertools import chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to video file.")
    parser.add_argument("--output_directory", type=str,
                        help="Path to directory where output mosaics are to be saved. Default is invokation path.",
                        default=".")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start_time", type=float, help="Time (secs) to start from.")
    group.add_argument("--start_frame", type=int, help="Frame number to start from.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--finish_time", type=float, help="Time (secs) to finish at.")
    group.add_argument("--finish_frame", type=int, help="Frame number to finish at.")
    parser.add_argument("--frame_skip", type=int, default=None,
                        help="Number of frames to skip between each mosaic update.")
    parser.add_argument("--orientation_file", type=str, default=None,
                        help="Path to .csv file containing orientation measurements.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sync_points", type=float, nargs=2, default=None,
                       help="Time points (sec) where video and orientation file are in sync, used to calculate time offset between video and timestamps in orientation file.")
    group.add_argument("--time_offset", type=float, default=None,
                       help="Time offset (sec) between video and orientation file timestamps, used for synchronization.")
    parser.add_argument("--min_matches", type=int, default=4,
                        help="Minimum number of matches to proceed with registration.")
    parser.add_argument("--min_features", type=int, default=100,
                        help="Minimum number of features to detect in an image.")
    parser.add_argument("--max_warp_size", type=int, nargs='+', default=None,
                        help="Maximum size of warped image (used to prevent OOM errors), if 1 argument given then image is clipped to square, if 2 then the order is height, width.")
    parser.add_argument("--max_mosaic_size", type=int, default=None,
                        help="Largest allowable size (width or height) for mosaic. Creates a new tile if it gets bigger.")
    parser.add_argument("--save_freq", type=int, default=0,
                        help="Save frequency for output mosaic (if less than 1 then output saves at exit).")
    parser.add_argument("--scale_factor", type=float, default=0.0,
                        help="Scale the input image with constant aspect ratio.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha blending scalar for merging new frames into mosaic.")
    parser.add_argument("--show_rotation", action="store_true",
                        help="Flag to display the rotation compensation using rotation data.")
    parser.add_argument("--show_mosaic", action="store_true", help="Flag to display the mosaic output.")
    parser.add_argument("--show_preprocessing", action="store_true", help="Flag to display the preprocessed image")
    parser.add_argument("--fix_color", action="store_true", help="Flag to preprocess image for color balance.")
    parser.add_argument("--fix_contrast", action="store_true",
                        help="Flag to preprocess image for contrast equalization.")
    parser.add_argument("--fix_light", action="store_true", help="Flag to preprocess image for lighting equalization.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--calibration", type=str, default=None, help="Path to calibration file.")
    group.add_argument("-k", "--intrinsic", nargs=9, type=float, default=None,
                       help="Space delimited list of intrinsic matrix terms, Read as K[0,0],K[1,0],K[2,0],K[1,0],K[1,1],K[1,2],K[2,0],K[2,1],K[2,2]")
    parser.add_argument("-d", "--distortion", nargs="+", type=float, default=None,
                        help="Space delimited list of distortion coefficients, Read as K1, K2, p1, p2")
    parser.add_argument("-x", "--xrotation", type=float, default=0,
                        help="Rotation around image plane's x axis (radians).")
    parser.add_argument("-y", "--yrotation", type=float, default=0,
                        help="Rotation around image plane's y axis (radians).")
    parser.add_argument("-z", "--zrotation", type=float, default=0,
                        help="Rotation around image plane's z axis (radians).")
    parser.add_argument("-g", "--gradientclip", type=float, default=0,
                        help="Clip the gradient of severely distorted image.")
    parser.add_argument("-f", "--fisheye", action="store_true", help="Flag to use fisheye distortion model.")
    parser.add_argument("--homography", type=str, choices=["similar", "affine", "perspective"], default="similar", help="Type of 2D homography to perform.")
    group = parser.add_argument_group()
    group.add_argument("--demo", action="store_true", help="Creates a video of the mosaic creation process. For demo purposes only.")
    group.add_argument("--show_demo", action="store_true", help="Display the demo while underway.")
    parser.add_argument("--features", type=str, nargs="+", choices=["ORB", "SIFT", "SURF", "BRISK", "KAZE", "ALL"], default="ALL", help="Set of features to use in registration.")
    parser.add_argument("--show_matches", action="store_true", help="Display the matches.")
    parser.add_argument("--inliers_roi", action="store_true", help="Only allow the convex hull of the inlier points to be displayed.")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    assert video_path.exists(), "File not found: {}".format(str(video_path))
    output_path = Path(args.output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if args.orientation_file is not None:
        ori_path = Path(args.orientation_file).resolve()
        assert ori_path.exists(), "File not found: {}".format(str(ori_path))
        if args.sync_points is None and args.time_offset is None:
            sys.stderr.write(
                "Warning: No --sync_points or --time_offset argument given, assuming video and orientation file start at the same time.\n")
        orientations = utils.load_orientations(ori_path, args)
        quat_lut = interp1d(orientations.index, orientations[["qx", "qy", "qz", "qw"]], axis=0)

    reader = cv2.VideoCapture(str(video_path))
    reader = utils.get_starting_pos(reader, args)
    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)) if args.scale_factor is None else int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)*args.scale_factor)
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)) if args.scale_factor is None else int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)*args.scale_factor)

    if args.demo:
        output_video = output_path.joinpath("demo.mp4")
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, frameSize=(1920, 1080), isColor=True)
    if args.show_demo:
        cv2.namedWindow("DEMO", cv2.WINDOW_NORMAL)

    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    formatspec = "{:0" + "{}d".format(len(str(n_frames))) + "}"
    if args.show_mosaic:
        cv2.namedWindow(str(video_path), cv2.WINDOW_NORMAL)
    if args.show_rotation:
        cv2.namedWindow("ROTATION COMPENSATION", cv2.WINDOW_NORMAL)
    if args.show_preprocessing:
        cv2.namedWindow("PREPROCESSING RESULT", cv2.WINDOW_NORMAL)
    if args.show_matches:
        cv2.namedWindow("MATCHING RESULT", cv2.WINDOW_NORMAL)

    if "ALL" in args.features:
        models = ["ORB", "SIFT", "SURF", "BRISK", "KAZE"]
    else:
        models = args.features
    detectors = []
    for model in models:
        if model == "ORB":
            detectors.append(cv2.ORB_create())  # ORB detector is pretty good and is CC licensed
        if model == "SIFT":
            detectors.append(cv2.SIFT_create())
        if model == "SURF":
            try:
                detectors.append(cv2.xfeatures2d.SURF_create())
            except cv2.error:
                sys.stderr.write("WARNING: Trying to use non-free SURF on OpenCV built with non-free option disabled.\n")
        # if model == "FAST":
        #     detectors.append(cv2.FastFeatureDetector_create())
        # if model == "BRIEF":
        #    detectors.append(cv2.xfeatures2d.BriefDescriptorExtractor_create())
        # if model == "FREAK":
        #    detectors.append(cv2.xfeatures2d.FREAK_create())
        # if model == "GFTT":
        #    detectors.append(cv2.GFTTDetector_create())
        if model == "BRISK":
            detectors.append(cv2.BRISK_create())
        if model == "KAZE":
            detectors.append(cv2.KAZE_create())


    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Camera Intrinsic Matrix
    if args.intrinsic is not None:
        K = np.array(args.intrinsic).reshape((3, 3)).T
    elif args.calibration is not None:
        K = np.eye(3)
        K[0, 2] = float(width) / 2
        K[1, 2] = float(height) / 2
    else:
        K = np.eye(3)
        K[0, 2] = float(width) / 2
        K[1, 2] = float(height) / 2
    print("K: {}".format(repr(K)))

    # Camera Lens distortion coefficients
    distCoeff = np.zeros((4, 1), np.float64)
    if args.distortion is not None:
        distCoeff = np.array([[d] for d in args.distortion], np.float64)

    # BEGIN MAIN LOOP #
    init = True  # Flag to init
    nomatches = False  # Prevents consecutive tile dumps due to no match
    counter = 0
    tile_counter = 0
    try:  # External try to handle any unexpected errors
        # Loop until stopping criteria is reached
        while not utils.evaluate_stopping(reader, args):
            # Print out the frame number
            sys.stdout.write("Processing Frame {}\n".format(int(reader.get(cv2.CAP_PROP_POS_FRAMES) + 1)))
            if args.frame_skip is not None and not init:
                reader.set(cv2.CAP_PROP_POS_FRAMES, reader.get(cv2.CAP_PROP_POS_FRAMES) + args.frame_skip - 1)

            # Acquire a frame
            ret, img = reader.read()
            og = img.copy()  # keep a copy for later reference

            if not ret:
                sys.stderr.write(
                    "Frame missing: {}\n".format(formatspec.format(int(reader.get(cv2.CAP_PROP_POS_FRAMES)))))
                continue

            # Preprocess the image
            # First rectify the image
            if args.fisheye:
                img = fisheye.undistortImage(img, K, distCoeff)
                image_mask = fisheye.undistortImage(255*np.ones_like(img), K, distCoeff)[:,:,0]
            else:
                img = cv2.undistort(img, K, distCoeff)
                image_mask = cv2.undistort(255 * np.ones_like(img), K, distCoeff)[:,:,0]
            # Then apply color correction if specified
            img = preprocessing.fix_color(img) if args.fix_color else img
            # Then apply contrast balancing if specified
            img = preprocessing.fix_contrast(img) if args.fix_contrast else img
            # Then apply light balancing if specified
            img = preprocessing.fix_light(img) if args.fix_light else img
            # Enhance detail
            img = preprocessing.enhance_detail(img)

            # DEBUGGING: DISPLAY THE PREPROCESSING
            if args.show_preprocessing:
                cv2.imshow("PREPROCESSING RESULT", np.concatenate((og, img), axis=1))

            # If it's the first time, then acquire keypoints and go to next frame
            if init:
                # Detect keypoints on the first frame
                features = [registration.get_features(img, detector) for detector in detectors]
                kp_prev = list(chain.from_iterable([f[0] for f in features]))
                num_features = len(kp_prev)
                des_prev = [f[1] for f in features]
                if num_features < args.min_features:
                    sys.stderr.write("Not Enough Features, Finding Good Frame.\n")
                    continue
                # Apply scaling to image if specified
                if args.scale_factor > 0.0:
                    img, kp_prev, image_mask = transformations.apply_scale(img, kp_prev, args.scale_factor, image_mask)
                # Apply rotation to image if necessary
                if args.orientation_file is not None:
                    lookup_time = reader.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    R = Rotation.from_quat(quat_lut(lookup_time))
                    img, image_mask, kp_prev = transformations.apply_transform(img, K, R, np.zeros(3), kp_prev,
                                                                               gradient_clip=args.gradientclip,
                                                                               mask=image_mask)
                    if args.show_rotation:
                        cv2.imshow("ROTATION COMPENSATION", img)
                else:
                    R = Rotation.from_euler("xyz", [args.xrotation, args.yrotation, args.zrotation])
                    img, image_mask, kp_prev = transformations.apply_transform(img, K, R, np.zeros(3), kp_prev,
                                                                               gradient_clip=args.gradientclip,
                                                                               mask=image_mask)
                # Update K to center on the image
                K[:2,-1] = [img.shape[1]/2, img.shape[0]/2]
                mosaic_mask = image_mask.copy()
                mosaic_img = img.copy()  # initialize the mosaic
                prev_img = img.copy()  # store the image as previous
                init = False
                A = None
                t = [0, 0]
                continue
            else:
                # Detect keypoints in the new frame
                features = [registration.get_features(img, detector) for detector in detectors]
                kp = list(chain.from_iterable([f[0] for f in features]))
                num_features = len(kp)
                des = [f[1] for f in features]
                if num_features < args.min_features:
                    sys.stderr.write("Not Enough Features, Skipping Frame.\n")
                    continue
                if args.scale_factor > 0.0:
                    img, kp, image_mask = transformations.apply_scale(img, kp, args.scale_factor, image_mask)
                # TODO Warp the images according to the camera transform
                # Apply rotation to image if necessary
                if args.orientation_file is not None:
                    lookup_time = reader.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    R = Rotation.from_quat(quat_lut(lookup_time))
                    img, image_mask, kp = transformations.apply_transform(img, K, R, np.zeros(3), kp,
                                                                               gradient_clip=args.gradientclip,
                                                                               mask=image_mask)
                    if args.show_rotation:
                        cv2.imshow("ROTATION COMPENSATION", img)
                else:
                    R = Rotation.from_euler("xyz", [args.xrotation, args.yrotation, args.zrotation])
                    img, image_mask, kp = transformations.apply_transform(img, K, R, np.zeros(3), kp,
                                                                               gradient_clip=args.gradientclip,
                                                                               mask=image_mask)
                # Update K to center on the image
                K[:2, -1] = [img.shape[1] / 2, img.shape[0] / 2]
                if args.show_rotation:
                    cv2.imshow("ROTATION COMPENSATION", img)

            ret, matches = registration.get_matches(des_prev, des, flann, args.min_matches)

            if not ret:
                sys.stderr.write("Not enough matches, starting new mosaic.\n")
                init = True
                if not nomatches:
                    nomatches = True
                    sys.stdout.write("Cropping tile.\n")
                    tile_counter = tile_counter + 1  # Increase the tile counter
                    fpath = output_path.joinpath("tile_{:03d}.png".format(tile_counter))  # Path for output tile
                    cv2.imwrite(str(fpath), mosaic_img)  # Save the output tile
                continue
            nomatches = False
            # Current Image Keypoints
            src_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            if args.inliers_roi:
                image_mask = preprocessing.convex_mask(img, src_pts)

            # Previous Image Keypoints
            dst_pts = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            # Warp the previous image keypoints onto the mosaic
            if A is not None:
                A = np.concatenate((A, np.array([[0, 0, 1]])), axis=0) if A.size < 9 else A
                dst_pts = cv2.perspectiveTransform(dst_pts, A)
            # Get the new affine alignment and bounds
            # TODO Sometimes homography comes and wraps everything up, apparently because of vanishing point. Since I cannot
            #  estimate vanishing point easily, I might have to live with it.
            A, xbounds, ybounds = registration.get_alignment(src_pts, img.shape[:2], dst_pts, mosaic_img.shape[:2],
                                                             homography=args.homography, gradient=args.gradientclip)
            # Get the C of the top left corner of the image to be inserted
            t = [-min(xbounds), -min(ybounds)]

            # This checks if the mosaic has reached a maximum size in either width or height dimensions.
            if args.max_mosaic_size is not None:
                if max(xbounds)-min(xbounds) > args.max_mosaic_size or max(ybounds)-min(ybounds) > args.max_mosaic_size:
                    sys.stdout.write("Max Size Reached, Cropping tile.\n")
                    tile_counter = tile_counter + 1  # Increase the tile counter
                    fpath = output_path.joinpath("tile_{:03d}.png".format(tile_counter))  # Path for output tile
                    cv2.imwrite(str(fpath), mosaic_img)  # Save the output tile
                    # First, isolate the image
                    bounds = cv2.boundingRect(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
                    mosaic_img = warped[bounds[1]:bounds[1]+bounds[3],
                                 bounds[0]:bounds[0]+bounds[2], :]
                    mosaic_mask = mosaic_mask[bounds[1]:bounds[1]+bounds[3],
                                 bounds[0]:bounds[0]+bounds[2]]
                    # Second, store the keypoints referenced to the new mosaic
                    points = [tuple(np.array(k.pt) - bounds[:2]) for k in kp]
                    for k,p in zip(kp,points):
                        k.pt = p
                    kp_prev = kp
                    des_prev = [d.copy() for d in des]
                    # Third, reset the affine homography
                    A = None
                    continue

            # warp the input image into the mosaic's plane
            if args.homography in ["similar", "affine"]:
                warped = cv2.warpAffine(img, A[:2,:], (max(xbounds)-min(xbounds), max(ybounds)-min(ybounds)))
                warped_mask = cv2.warpAffine(image_mask, A[:2,:], (max(xbounds) - min(xbounds), max(ybounds) - min(ybounds)))
            else:
                warped = cv2.warpPerspective(img, A, (max(xbounds)-min(xbounds), max(ybounds)-min(ybounds)))
                warped_mask = cv2.warpPerspective(image_mask, A, (max(xbounds) - min(xbounds), max(ybounds) - min(ybounds)))
            warped_mask = (warped_mask == 255).astype(np.uint8) * 255

            # Get the previous iteration mosaic_mask in the shape of the update
            # First construct a template in the shape of the transformed image
            template = np.zeros_like(warped)
            # Get a one channel mask of that template
            mosaic_mask_ = template[:, :, 0].copy()
            # Insert the previous mosaic mask into the template mask
            mosaic_mask_[t[1]:mosaic_mask.shape[0] + t[1], t[0]:mosaic_mask.shape[1] + t[0]] = mosaic_mask
            # Copy the template into the mosaic placeholder
            mosaic_img_ = template.copy()
            # Insert the previous mosaic into the placeholder
            mosaic_img_[t[1]:mosaic_img.shape[0] + t[1], t[0]:mosaic_img.shape[1] + t[0]] = mosaic_img
            # Get the mask where mosaic and warped input intersect
            shared = cv2.bitwise_and(warped_mask, mosaic_mask_)
            # Combine the image and tile, and update
            # Insert the warped image into the mosaic placeholder
            mosaic_img = np.where(cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR) > 0, warped, mosaic_img_)
            # Create an alpha blend of the warped and the previous mosaic image together
            mixer = np.uint8(
                args.alpha * mosaic_img_.astype(np.float32) + (1.0 - args.alpha) * warped.astype(np.float32))
            # Insert the blending at the intersection only
            mosaic_img = np.where(cv2.cvtColor(shared, cv2.COLOR_GRAY2BGR) > 0, mixer, mosaic_img)
            # update the tile_mask with the warped mask region
            mosaic_mask = cv2.bitwise_or(mosaic_mask_, warped_mask)

            mosaic_img, crop = preprocessing.crop_to_valid_area(mosaic_img)
            mosaic_mask, crop = preprocessing.crop_to_valid_area(mosaic_mask)
            A[:2, -1] = A[:2, -1] - crop[:2]

            # Display the mosaic
            if args.show_mosaic:
                cv2.imshow(str(video_path), mosaic_img)

            frame = utils.prepare_frame(img, mosaic_img, (1920, 1080))
            if args.show_demo:
                cv2.imshow("DEMO", frame)
            if args.demo:
                writer.write(frame)

            prev_img = img.copy()  # Update the previous frame
            kp_prev = kp  # Update the previous keypoints
            des_prev = [d.copy() for d in des]  # Update the previous descriptors

            key = cv2.waitKey(1)
            counter = counter + 1
            if args.save_freq > 0 and counter % args.save_freq == 0:
                counter = 0
                fpath = output_path.joinpath("current_mosaic.png")
                cv2.imwrite(str(fpath), mosaic_img)
            if key == ord("q"):
                sys.stdout.write("Quitting.\n")
                break
    except Exception as err:
        # Some strange error has occurred, write out to stderr, cleanup and rethrow the error
        sys.stderr.write("\nPipeline failed at frame {}\n".format(reader.get(cv2.CAP_PROP_POS_FRAMES) + 1))
        cv2.destroyAllWindows()
        reader.release()
        if args.demo:
            writer.release()
        fpath = output_path.joinpath("error_frame.png")
        cv2.imwrite(str(fpath), img)
        fpath = output_path.joinpath("error_mosaic.png")
        cv2.imwrite(str(fpath), mosaic_img)
        raise err
    # The video exited properly, so cleanup and exit.
    fpath = output_path.joinpath("tile_{:03d}.png".format(tile_counter))
    cv2.imwrite(str(fpath), mosaic_img)
    cv2.destroyAllWindows()
    reader.release()
    if args.demo:
        writer.release()


if __name__=="__main__":
    main()
