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
import yaml


def main():
    args = utils.parse_args()

    reader = utils.VideoPlayer(args)

    output_path = Path(args.output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if args.orientation_file is not None:
        ori_path = Path(args.orientation_file).resolve()
        assert ori_path.exists(), "File not found: {}".format(str(ori_path))
        if args.sync_points is None and args.time_offset is None:
            sys.stderr.write(
                "Warning: No --sync_points or --time_offset argument given, assuming video and orientation file start at the same time.\n")
        orientations = utils.load_orientations(ori_path, args)
        quat_lut = interp1d(orientations.index, orientations[["qx", "qy", "qz", "qw"]], axis=0, kind='nearest')

    # IF DEMO SPECIFIED, THEN GENERATE THE DEMO VIDEO
    if args.demo:
        output_video = output_path.joinpath("demo.mp4")
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), reader.fps, frameSize=(1920, 1080), isColor=True)

    # FORMAT SPEC FOR PROGRESS
    formatspec = "{:0" + "{}d".format(len(str(reader.n_frames))) + "}"

    # DISPLAY WINDOWS
    if args.show_mosaic:
        cv2.namedWindow(args.video, cv2.WINDOW_NORMAL)
    if args.show_rotation:
        cv2.namedWindow("ROTATION COMPENSATION", cv2.WINDOW_NORMAL)
    if args.show_preprocessing:
        cv2.namedWindow("PREPROCESSING RESULT", cv2.WINDOW_NORMAL)
    if args.show_matches:
        cv2.namedWindow("MATCHING RESULT", cv2.WINDOW_NORMAL)
    if args.show_demo:
        cv2.namedWindow("DEMO", cv2.WINDOW_NORMAL)

    # DEFINE FEATURE DETECTOR
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

    # DEFINE THE MATCHER METHOD
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # DEFINE THE CAMERA PROPERTIES
    # Camera Intrinsic Matrix
    # Default is set the calibration matrix to an identity matrix with transpose components centred on the image center
    K = np.eye(3)
    K[0, 2] = float(reader.width) / 2
    K[1, 2] = float(reader.height) / 2

    if args.intrinsic is not None:
        K = np.array(args.intrinsic).reshape((3, 3))  # If -k argument is defined, generate the K matrix

    if args.calibration is not None:
        # If a calibration file has been given (a ROS camera_info yaml style file)
        calibration_path = Path(args.calibration).resolve()
        assert calibration_path.exists(), "File not found: {}".format(str(calibration_path))
        with open(args.calibration, "r") as f:
            calib_data = yaml.safe_load(f)
        if 'camera_matrix' in calib_data:
            K = np.array(calib_data['camera_matrix']['data']).reshape((3, 3))
        else:
            sys.stderr.write(f"WARNING: No camera_matrix found in {str(calibration_path)}\n")

    print("K: {}".format(repr(K)))
    K_scaled = K.copy()  # A copy of K that is scaled

    # Camera Lens distortion coefficients
    distCoeff = np.zeros((4, 1), np.float64)
    if args.distortion is not None:
        distCoeff = np.array([[d] for d in args.distortion], np.float64)
    if args.calibration is not None:
        if 'distortion_coefficients' in calib_data:
            distCoeff = np.array(calib_data['distortion_coefficients']['data']).reshape((calib_data['distortion_coefficients']['rows'],
                                                                                         calib_data['distortion_coefficients']['cols']))
        else:
            sys.stderr.write(f"WARNING: No distortion_coefficients found in {str(calibration_path)}\n")

    # BEGIN MAIN LOOP #
    init = True  # Flag to init
    nomatches = False  # Prevents consecutive tile dumps due to no match
    counter = 0  # Counter to intermittently save the output mosaic
    tile_counter = 0  # Counter for number of generated tiles.
    try:  # External try to handle any unexpected errors
        print(f"Frames available: {len(reader)}")
        print(reader)
        # Loop through until a stopping condition is reached or a frame fails to return
        for img in reader:
            print(reader)
            og = img.copy()  # keep a copy for later reference

            # Preprocess the image
            # First rectify the image
            if args.fisheye:
                img = fisheye.undistortImage(img, K, distCoeff)
                image_mask = fisheye.undistortImage(255 * np.ones_like(img), K, distCoeff)[:, :, 0]
            else:
                img = cv2.undistort(img, K, distCoeff)
                image_mask = cv2.undistort(255 * np.ones_like(img), K, distCoeff)[:, :, 0]
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
                fontsize = 3
                border = np.zeros((og.shape[0], 20, 3), dtype=np.uint8)
                border[:, :, -1] = 255
                preproc_result = np.concatenate((og, border, img), axis=1)
                text_shape, baseline = cv2.getTextSize("BEFORE", cv2.FONT_HERSHEY_PLAIN, fontsize, 3)
                preproc_result = cv2.putText(preproc_result, "BEFORE", (10, 10 + text_shape[1]), cv2.FONT_HERSHEY_PLAIN,
                                             fontsize, (255, 255, 255), 3)
                text_shape, baseline = cv2.getTextSize("AFTER", cv2.FONT_HERSHEY_PLAIN, fontsize, 3)
                preproc_result = cv2.putText(preproc_result, "AFTER", (og.shape[1] + 20 + 10, 10 + text_shape[1]),
                                             cv2.FONT_HERSHEY_PLAIN,
                                             fontsize, (255, 255, 255), 3)
                cv2.imshow("PREPROCESSING RESULT", preproc_result)

            # If it's the first time, then acquire keypoints and go to next frame
            if init:
                sys.stdout.write("Initializing new mosaic.\n")
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
                    img, kp_prev, image_mask = transformations.apply_scale(img, kp_prev, args.scale_factor,
                                                                           image_mask)
                    K_scaled = K.copy()
                    K_scaled = K_scaled * args.scale_factor
                    K_scaled[2, 2] = 1
                # Apply rotation to image if necessary
                if args.orientation_file is not None:
                    lookup_time = reader.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    R = Rotation.from_quat(quat_lut(lookup_time))
                    # This is infinite homography
                    img, image_mask, kp_prev = transformations.apply_transform(img, K_scaled, R, np.zeros(3),
                                                                               kp_prev,
                                                                               gradient_clip=args.gradientclip,
                                                                               mask=image_mask)
                    if args.show_rotation:
                        cv2.imshow("ROTATION COMPENSATION", img)
                # If there is no orientation file specifying the world to camera transform, then rotate according to xyz arguments
                else:
                    R = Rotation.from_euler("xyz", [args.xrotation, args.yrotation, args.zrotation])
                    img, image_mask, kp_prev = transformations.apply_transform(img, K_scaled, R, np.zeros(3),
                                                                               kp_prev,
                                                                               gradient_clip=args.gradientclip,
                                                                               mask=image_mask)
                mosaic_mask = image_mask.copy()  # Initialize the mosaic mask
                mosaic_img = img.copy()  # initialize the mosaic
                prev_img = img.copy()  # store the image as previous
                A = None  # Initialize the affine aggregated homography as None so that it isn't applied straight away
                t = [0, 0]  # Initialize the translation to 0
                init = False  # Initialization complete
                sys.stdout.write("Init stage complete.\n")
                continue
            # We are now mosaicking
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
                    K_scaled = K.copy()
                    K_scaled = K_scaled * args.scale_factor
                    K_scaled[2, 2] = 1
                # Apply rotation to image if necessary
                if args.orientation_file is not None:
                    lookup_time = reader.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    R = Rotation.from_quat(quat_lut(lookup_time))
                    img, image_mask, kp = transformations.apply_transform(img, K_scaled, R, np.zeros(3), kp,
                                                                          gradient_clip=args.gradientclip,
                                                                          mask=image_mask)
                    if args.show_rotation:
                        cv2.imshow("ROTATION COMPENSATION", img)
                else:
                    R = Rotation.from_euler("xyz", [args.xrotation, args.yrotation, args.zrotation])
                    img, image_mask, kp = transformations.apply_transform(img, K_scaled, R, np.zeros(3), kp,
                                                                          gradient_clip=args.gradientclip,
                                                                          mask=image_mask)
                if args.show_rotation:
                    cv2.imshow("ROTATION COMPENSATION", img)
                # Compute matches between previous frame and current frame
                ret, matches = registration.get_matches(des_prev, des, flann, args.min_matches)
                # Handle not enough matches
                if not ret:
                    # We essentially start a new mosaic
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

                if args.show_matches:
                    img_kp = cv2.drawMatches(prev_img, kp_prev, img, kp, matches, np.zeros_like(prev_img))
                    cv2.imshow("MATCHING RESULT", img_kp)

                # Warp the previous image keypoints onto the mosaic, if the Affine homography is available.
                if A is not None:
                    A = np.concatenate((A, np.array([[0, 0, 1]])), axis=0) if A.size < 9 else A
                    dst_pts = cv2.perspectiveTransform(dst_pts, A)
                # E, _ = cv2.findEssentialMat(src_pts, dst_pts, K_scaled)
                # _, pose_R, pose_t, _ = cv2.recoverPose(E, src_pts, dst_pts, K_scaled)
                # pose_euler = Rotation.from_matrix(pose_R).as_euler("xyz", degrees=True)
                # print("Pose\n\tRotation:\n\t\tRoll: {:.2f}\tPitch: {:.2f}\tYaw: {:.2f}\n\t"
                #       "Translation:\n\t\tX: {:.2f}\tY: {:.2f}\tZ: {:.2f}".format(*pose_euler, *pose_t.squeeze()))
                # Get the new affine alignment and bounds
                # TODO Sometimes homography wraps the image around, apparently because of vanishing point and the horizon.
                #  I cannot estimate the vanishing point easily, so in this iteration we have to live with this problem.
                #  A way to address this is to mask out the components of the image that do not intersect with the ground plane.
                #  We would need to estimate the ground plane, and compute the intersection of camera rays with this plane.
                #  If they intersect, then the pixel that the ray goes through needs to be preserved, others are discarded.
                A, xbounds, ybounds = registration.get_alignment(src_pts, img.shape[:2], dst_pts, mosaic_img.shape[:2],
                                                                 homography=args.homography, gradient=args.gradientclip)
                # Get the C of the top left corner of the warped image to be inserted
                t = [-min(xbounds), -min(ybounds)]

                # TODO: Handling the tiles needs an overhaul.
                #  I would like to track the coordinates of the mosaic, and assign tiles to grid on this coordinate system.
                #  The warped image should not need to be padded. Instead the warped image pixels should be assigned a
                #  coordinated within the mosaic coordinate system. Then, the tiles that share coordinates from the warped
                #  image are sequentially updated with the pixels.
                # This checks if the mosaic has reached a maximum size in either width or height dimensions.
                if args.max_mosaic_size is not None:
                    if max(xbounds) - min(xbounds) > args.max_mosaic_size or max(ybounds) - min(
                            ybounds) > args.max_mosaic_size:
                        sys.stdout.write("Max Size Reached, Cropping tile.\n")
                        tile_counter = tile_counter + 1  # Increase the tile counter
                        fpath = output_path.joinpath("tile_{:03d}.png".format(tile_counter))  # Path for output tile
                        cv2.imwrite(str(fpath), mosaic_img)  # Save the output tile
                        # First, isolate the image
                        bounds = cv2.boundingRect(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
                        mosaic_img = warped[bounds[1]:bounds[1] + bounds[3],
                                     bounds[0]:bounds[0] + bounds[2], :]
                        mosaic_mask = mosaic_mask[bounds[1]:bounds[1] + bounds[3],
                                      bounds[0]:bounds[0] + bounds[2]]
                        # Second, store the keypoints referenced to the new mosaic
                        points = [tuple(np.array(k.pt) - bounds[:2]) for k in kp]
                        for k, p in zip(kp, points):
                            k.pt = p
                        kp_prev = kp
                        des_prev = [d.copy() for d in des]
                        # Third, reset the affine homography
                        A = None
                        continue

                # warp the input image into the mosaic's plane
                if args.homography in ["similar", "affine"]:
                    warped = cv2.warpAffine(img, A[:2, :], (max(xbounds) - min(xbounds), max(ybounds) - min(ybounds)))
                    warped_mask = cv2.warpAffine(image_mask, A[:2, :],
                                                 (max(xbounds) - min(xbounds), max(ybounds) - min(ybounds)))
                else:
                    warped = cv2.warpPerspective(img, A, (max(xbounds) - min(xbounds), max(ybounds) - min(ybounds)))
                    warped_mask = cv2.warpPerspective(image_mask, A,
                                                      (max(xbounds) - min(xbounds), max(ybounds) - min(ybounds)))
                warped_mask = (warped_mask == 255).astype(np.uint8) * 255

                # Get the previous iteration mosaic_mask in the shape of the update
                # Get a one channel mask of that template
                mosaic_mask_ = np.zeros_like(warped_mask)
                # Insert the previous mosaic mask into the template mask
                mosaic_mask_[t[1]:mosaic_mask.shape[0] + t[1], t[0]:mosaic_mask.shape[1] + t[0]] = mosaic_mask
                # Copy the template into the mosaic placeholder
                mosaic_img_ = np.zeros_like(warped)
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

                # Display the mosaic
                if args.show_mosaic:
                    cv2.imshow(args.video, mosaic_img)

                if args.show_demo or args.demo:
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
        del reader
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
    del reader
    if args.demo:
        writer.release()


if __name__=="__main__":
    main()
