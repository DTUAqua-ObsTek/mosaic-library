import itertools
import json
import pickle
from abc import ABC, abstractmethod
from os import PathLike
from typing import Union, AnyStr, Sequence, Any

import cv2
import mosaicking
from cv2 import fisheye
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
import warnings
import traceback

import mosaicking.transformations
from mosaicking import preprocessing, utils, transformations, registration, core
import yaml

import networkx as nx
from dataclasses import asdict

from shapely import geometry


def main():
    args = utils.parse_args()

    reader = utils.VideoPlayer(args)

    output_path = Path(args.output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if args.orientation_file is not None:
        ori_path = Path(args.orientation_file).resolve()
        assert ori_path.exists(), "File not found: {}".format(str(ori_path))
        if args.sync_points is None and args.time_offset is None:
            warnings.warn("No --sync_points or --time_offset argument given, assuming video and orientation file start at the same time.", UserWarning)
        orientations = utils.load_orientations(ori_path, args)
        quat_lut = interp1d(orientations.index, orientations[["qx", "qy", "qz", "qw"]], axis=0, kind='nearest')

    # IF DEMO SPECIFIED, THEN GENERATE THE DEMO VIDEO
    if args.demo:
        output_video = output_path.joinpath("demo.mp4")
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), reader.fps, frameSize=(1920, 1080), isColor=True)

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
            detectors.append(registration.OrbDetector())  # ORB detector is pretty good and is CC licensed
        if model == "SIFT":
            detectors.append(registration.SiftDetector())  # SIFT detector is powerful but research use only
        if model == "SURF":
            try:
                detectors.append(registration.SurfDetector())
            except cv2.error:
                warnings.warn("Trying to use non-free SURF on OpenCV built with non-free option disabled.", UserWarning)
        if model == "FAST":
            warnings.warn("FAST features are not yet implemented.", UserWarning)
            #detectors.append(cv2.FastFeatureDetector_create())
        if model == "BRIEF":
            warnings.warn("BRIEF features are not yet implemented.", UserWarning)
            #detectors.append(cv2.xfeatures2d.BriefDescriptorExtractor_create())
        if model == "FREAK":
            warnings.warn("FREAK features are not yet implemented.", UserWarning)
            #detectors.append(cv2.xfeatures2d.FREAK_create())
        if model == "GFTT":
            warnings.warn("GFTT features are not yet implemented.", UserWarning)
            #detectors.append(cv2.GFTTDetector_create())
        if model == "BRISK":
            detectors.append(registration.BriskDetector())
        if model == "KAZE":
            detectors.append(registration.KazeDetector())
        if model == "AKAZE":
            detectors.append(registration.AkazeDetector())

    matcher = registration.Matcher()

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
            warnings.warn(f"No camera_matrix found in {str(calibration_path)}", UserWarning)

    # Camera Lens distortion coefficients
    dist_coeff = np.zeros((4, 1), np.float64)
    if args.distortion is not None:
        dist_coeff = np.array([[d] for d in args.distortion], np.float64)
    if args.calibration is not None:
        if 'distortion_coefficients' in calib_data:
            dist_coeff = np.array(calib_data['distortion_coefficients']['data']).reshape((calib_data['distortion_coefficients']['rows'],
                                                                                          calib_data['distortion_coefficients']['cols']))
        else:
            warnings.warn(f"No distortion_coefficients found in {str(calibration_path)}", UserWarning)

    K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, (reader.width, reader.height), 0)

    print("K:", K)
    K_scaled = K.copy()  # A copy of K that is scaled

    # BEGIN MAIN LOOP #
    img = None
    mosaic_img = None
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
                img = fisheye.undistortImage(img, K, dist_coeff)
                image_mask = fisheye.undistortImage(255 * np.ones_like(img), K, dist_coeff)[:, :, 0]
            else:
                img = cv2.undistort(img, K, dist_coeff)
                image_mask = cv2.undistort(255 * np.ones_like(img), K, dist_coeff)[:, :, 0]
            # Then apply color correction if specified
            img = preprocessing.imadjust(img) if args.imadjust else img
            # Then apply contrast balancing if specified
            img = preprocessing.equalize_color(img) if args.equalize_color else img
            # Then apply light balancing if specified
            img = preprocessing.equalize_luminance(img) if args.equalize_luminance else img
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
                kp_prev, des_prev = registration.get_keypoints_descriptors(img, detectors)
                num_features = len(kp_prev)
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
                kp, des = registration.get_keypoints_descriptors(img, detectors)
                num_features = len(kp)
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
                ret, matches = registration.get_matches(des_prev, des, matcher, args.min_matches)
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
                A, xbounds, ybounds = mosaicking.transformations.get_alignment(src_pts, img.shape[:2], dst_pts, mosaic_img.shape[:2],
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
                if key == ord("q") or key & 0xff == 27:
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        sys.stderr.write("\nUser terminated program.\n")
    except Exception:
        # Some strange error has occurred, write out to stderr, cleanup and rethrow the error
        sys.stderr.write("\nPipeline failed at frame {}\n".format(reader.get(cv2.CAP_PROP_POS_FRAMES) + 1))
        traceback.print_exc()
        fpath = output_path.joinpath("error_frame.png")
        if img is not None:
            cv2.imwrite(str(fpath), img)
        fpath = output_path.joinpath("error_mosaic.png")
        if mosaic_img is not None:
            cv2.imwrite(str(fpath), mosaic_img)
    finally:
        # Cleanup actions and exit.
        fpath = output_path.joinpath("tile_{:03d}.png".format(tile_counter))
        if mosaic_img is not None:
            cv2.imwrite(str(fpath), mosaic_img)
        cv2.destroyAllWindows()
        reader.release()
        #del reader
        if args.demo:
            writer.release()


if __name__ == "__main__":
    main()

class Mapper:
    def __init__(self, output_width: int, output_height: int):
        self._canvas, self._canvas_mask = self._create_canvas(output_width, output_height)

    def _create_canvas(self, output_width: int, output_height: int) -> tuple[Union[np.ndarray, cv2.cuda.GpuMat],
                                                                       Union[np.ndarray, cv2.cuda.GpuMat]]:
        if mosaicking.HAS_CUDA:
            return self._cuda_create_canvas(output_width, output_height)
        return self._cpu_create_canvas(output_width, output_height)

    @staticmethod
    def _cuda_create_canvas(output_width: int, output_height: int) -> tuple[cv2.cuda.GpuMat,
                                                                      cv2.cuda.GpuMat]:
        tmp, tmp_mask = Mapper._cpu_create_canvas(output_width, output_height)
        output = cv2.cuda.GpuMat(tmp)
        output_mask = cv2.cuda.GpuMat(tmp_mask)
        return output, output_mask

    @staticmethod
    def _cpu_create_canvas(output_width: int, output_height: int) -> tuple[np.ndarray, np.ndarray]:
        output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        output_mask = np.zeros((output_height, output_width), dtype=np.uint8)
        return output, output_mask

    def _update_cuda(self, image: cv2.cuda.GpuMat, H: np.ndarray, stream: cv2.cuda.Stream = None):
        image = preprocessing.make_bgr(image)
        dsize = self._canvas_mask.size()
        width, height = image.size()
        mask = cv2.cuda.GpuMat(255 * np.ones((height, width), dtype=np.uint8))
        warped = cv2.cuda.warpPerspective(image, H, dsize, None, cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        warped_mask = cv2.cuda.warpPerspective(mask, H, dsize, None, cv2.INTER_CUBIC,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # Create binary masks for the regions
        output_mask_bin = cv2.cuda.threshold(self._canvas_mask, 1, 255, cv2.THRESH_BINARY)[1]
        warped_mask_bin = cv2.cuda.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)[1]

        # Identify the intersecting and exclusive regions
        mask_intersect = cv2.cuda.bitwise_and(output_mask_bin, warped_mask_bin)
        output_mask_only = cv2.cuda.bitwise_and(output_mask_bin, cv2.cuda.bitwise_not(warped_mask_bin))
        warped_mask_only = cv2.cuda.bitwise_and(warped_mask_bin, cv2.cuda.bitwise_not(output_mask_bin))

        # Copy the warped region to the exclusively warped region (that's it for now)
        warped.copyTo(warped_mask_only, self._canvas)
        # Update the output mask with the warped region mask
        warped_mask_only.copyTo(warped_mask_only, self._canvas_mask)

        # Blend the intersecting regions
        # Prepare an alpha blending mask
        alpha_gpu = cv2.cuda.normalize(mask_intersect, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F, cv2.cuda.GpuMat())
        # Alpha blend the intersecting region
        blended = alpha_blend_cuda(self._canvas, warped, alpha_gpu)
        # Convert to 8UC3
        blended = cv2.cuda.merge(tuple(
            cv2.cuda.normalize(channel, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_8U, cv2.cuda.GpuMat()) for channel in
            cv2.cuda.split(blended)), cv2.cuda.GpuMat())
        blended.copyTo(mask_intersect, self._canvas)

        # cleanup
        mask.release()
        warped.release()
        warped_mask.release()
        output_mask_bin.release()
        warped_mask_bin.release()
        mask_intersect.release()
        output_mask_only.release()
        warped_mask_only.release()
        alpha_gpu.release()
        blended.release()

    def _update_cpu(self, image: np.ndarray, H: np.ndarray):
        ...

    def update(self, image: Union[np.ndarray, cv2.cuda.GpuMat], H: np.ndarray, stream: cv2.cuda.Stream = None):
        if mosaicking.HAS_CUDA:
            if isinstance(image, np.ndarray):
                image = cv2.cuda.GpuMat(image.copy())
            self._update_cuda(image, H, stream)
        else:
            self._update_cpu(image, H)

    def release(self):
        if mosaicking.HAS_CUDA:
            self._canvas.release()
            self._canvas_mask.release()

    @property
    def output(self):
        return self._canvas.download()

class Mosaic(ABC):

    def __init__(self, data_path: Union[AnyStr, PathLike, Path] = None,
                 output_path: Union[AnyStr, PathLike, Path] = None,
                 feature_types: Sequence[str] = ('ORB',),
                 preprocessing_params: Sequence[tuple[str, dict[str, Any], dict[str, Any]]] = None,
                 intrinsics: dict[str, np.ndarray] = None,
                 orientation_path: Union[AnyStr, PathLike, Path] = None,
                 time_offset: float = 0.0,
                 verbose: bool = False,
                 caching: bool = True,
                 overwrite: bool = True,
                 force_cpu: bool = False,
                 player_params: dict[str, Any] = None,
                 ):
        assert data_path is not None or output_path is not None, "data_path and output_path cannot both be unspecified."
        assert not Path(output_path).is_file(), "output_path is not a directory."

        if output_path is not None:
            output_path = Path(output_path)  # convert output_path to pathlib.Path
        if data_path is not None:
            data_path = Path(data_path).resolve(True)  # convert data_path to pathlib.Path (must exist)
        if orientation_path is not None:
            orientation_path = Path(orientation_path).resolve(True)  # convert orientation_path to pathlib.Path (must exist)

        self._meta = None  # initialize metadata variable
        # output_path specified, either an old configuration to be rerun or a new output.
        if output_path is not None:
            # load meta if meta.json exists, then it is a rerun.
            if output_path.joinpath("meta.json").exists() and not overwrite:
                with output_path.joinpath("meta.json").open() as f:
                    self._meta = json.load(f, )
                # check data_path is correct
                assert "data_path" in self._meta, "meta missing attribute data_path."
                assert self._meta["data_path"] is not None, "meta attribute data_path undefined."
                if data_path is not None:
                    # Warn that data_path doesn't match meta.data_path
                    if self._meta["data_path"] != data_path:
                        warnings.warn(f"meta.data_path does not match data_path argument, using meta.data_path. "
                                      f"Call with overwrite argument to overwrite meta.data_path.")
            # if meta.json doesn't exist, make sure data_path does.
            else:
                assert data_path is not None, "data_path undefined and meta.json doesn't exist."
                self._meta = dict(data_path=data_path,
                                  output_path=output_path,
                                  feature_types=feature_types,
                                  preprocessing_params=preprocessing_params,
                                  intrinsics=intrinsics,
                                  orientation_path=orientation_path,
                                  time_offset=time_offset,
                                  force_cpu=force_cpu,
                                  player_params=player_params,)
        else:
            self._meta = dict(data_path=data_path,
                              output_path=data_path.with_name(data_path.stem + "_mosaic"),
                              feature_types=feature_types,
                              preprocessing_params=preprocessing_params,
                              intrinsics=intrinsics,
                              orientation_path=orientation_path,
                              time_offset=time_offset,
                              force_cpu=force_cpu,
                              player_params=player_params,)

        self._verbose = verbose                          # For logging purposes.
        self._reader_obj = self._create_reader_obj()     # For reading the data.
        self._registration_obj = core.ImageGraph()       # For registration tracking
        # For feature extraction
        self._feature_extractor = registration.CompositeDetector(self._meta["feature_types"], self._meta["force_cpu"])
        self._matcher = registration.CompositeMatcher()  # For feature matching
        self._preprocessor_pipeline, self._preprocessor_args = self._create_preprocessor_obj(preprocessing_params)
        self._caching = caching                          # Flag to cache feature extraction
        self._overwrite = overwrite                      # Flag to overwrite anything in output path
        # self._meta["output_path"].mkdir(parents=True, exist_ok=True)
        # if not self._meta["output_path"].joinpath("meta.json").exists() or self._overwrite:
        #     if self._overwrite:
        #         warnings.warn(f"Overwriting {self._meta['output_path'] / 'meta.json'}")
        #     with open(self._meta["output_path"] / "meta.json", "w") as f:
        #         json.dump(self._meta, f, cls=core.PathEncoder)

    @abstractmethod
    def _create_reader_obj(self) -> utils.DataReader:
        """
        A method to create the reader object from self._meta.
        """

    @staticmethod
    def _create_preprocessor_obj(preprocessing_params: Sequence[tuple[str, dict[str, Any], dict[str, Any]]]) -> tuple[preprocessing.Pipeline, Sequence[dict[str, Any]]]:
        if preprocessing_params is None:
            return preprocessing.Pipeline(tuple()), tuple()
        obj_strings, init_args, args = zip(*preprocessing_params)
        objs = preprocessing.parse_preprocessor_strings(*obj_strings)
        pipeline = preprocessing.Pipeline([o(**arg) for o, arg in zip(objs, init_args)])
        return pipeline, args

    def _load_features(self):
        cache_path = self._meta["output_path"].joinpath("features.pkl")
        assert cache_path.exists(), f"Features cache {cache_path} does not exist."
        with cache_path.open("rb") as f:
            features = pickle.load(f)
            self._features = tuple(utils.convert_feature_keypoints(feature) for feature in features)

    def _load_orientations(self) -> Union[None, Slerp]:
        orientations_path = self._meta["orientation_path"]
        time_offset = self._meta["time_offset"]
        if orientations_path is None or time_offset is None:
            return None
        assert orientations_path.exists(), f"Orientation path {orientations_path} does not exist."
        return utils.load_orientation(orientations_path, time_offset)

    @abstractmethod
    def extract_features(self):
        """
        A method for extracting features to self._features
        """
        ...

    @abstractmethod
    def match_features(self):
        """
        A method for matching features in self._features to self._matches
        """
        ...

    @abstractmethod
    def registration(self):
        """
        A method for registering transformations from matches in self._matches to self._transforms
        """
        ...

    @abstractmethod
    def generate(self):
        """
        A method for generating the output mosaics.
        """
        ...


class SequentialMosaic(Mosaic):
    # TODO: Add orientation adjustments
    # TODO: Add a self-healing strategy to find nice homographies between subgraphs.
    # TODO: Subdivide graph into tiles


    # Private methods
    def _create_reader_obj(self) -> utils.VideoPlayer:
        player_params = {} if self._meta["player_params"] is None else self._meta["player_params"]
        if mosaicking.HAS_CUDA:
            return utils.CUDAVideoPlayer(self._meta["data_path"], **player_params)
        return utils.CPUVideoPlayer(self._meta["data_path"], **player_params)

    def _close_video_reader(self):
        self._reader_obj.release()
        self._reader_obj = None

    # Required Overloads
    def extract_features(self):
        # TODO: support caching
        for frame_no, (ret, frame) in enumerate(self._reader_obj):
            if self._verbose:
                print(repr(self._reader_obj))
            # skip bad frame
            if not ret:
                self._registration_obj.add_node(frame_no)
                continue
            frame = preprocessing.make_gray(frame)  # Convert to grayscale
            frame = self._preprocessor_pipeline.apply(frame)  # Apply preprocessing to image
            data = core.ImageNode(self._feature_extractor.detect(frame))
            self._registration_obj.add_node(frame_no, **asdict(data))  # add image attributes to node

    def match_features(self):
        feature_types = self._meta["feature_types"]
        # Sequential matching
        num_nodes = len(self._registration_obj.nodes())
        for count, (node_idx, node_prev) in enumerate(self._registration_obj.nodes(data=True)):
            if count == num_nodes - 1:
                break
            node = self._registration_obj.nodes[node_idx + 1]
            # if node or node_prev don't have the features attribute, then skip (don't add edge)
            if 'features' not in node or 'features' not in node_prev:
                continue
            # Retrieve features that are non-empty
            descriptors_prev = {feature_type: features['descriptors'] for feature_type, features in node_prev['features'].items()}
            descriptors = {feature_type: features['descriptors'] for feature_type, features in node['features'].items()}
            all_matches = self._matcher.knn_match(descriptors, descriptors_prev)
            good_matches = dict()
            # Apply Lowe's distance ratio test to acquire good matches.
            for feature_type, matches in all_matches.items():
                good_matches.update({feature_type: tuple(m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance)})
            num_matches = sum([len(m) for m in good_matches.values()])
            # Perspective Homography requires at least 4 matches, if there aren't enough matches then don't create edge.
            # TODO: 10 is a magic number. Make it adjustable by user?
            if num_matches < 10:
                continue
            self._registration_obj.add_edge(node_idx, node_idx + 1, matches=good_matches)

    def registration(self):
        # Here, iterate through each subgraph and estimate homography.
        # If the homography quality is bad, then we prune the graph.
        to_filter = []
        for subgraph in (self._registration_obj.subgraph(c).copy() for c in nx.weakly_connected_components(self._registration_obj)):
            for node_prev, node, edge_data in subgraph.edges(data=True):
                features_prev = subgraph.nodes[node_prev]['features']
                features = subgraph.nodes[node]['features']
                matches = edge_data['matches']
                kp_prev, kp = [], []
                # Go through each feature type
                for feature_type in features:
                    # for each match, obtain the registered keypoints
                    for match in matches[feature_type]:
                        kp_prev.append(features_prev[feature_type]['keypoints'][match.trainIdx])
                        kp.append(features[feature_type]['keypoints'][match.queryIdx])
                kp_prev = np.stack(kp_prev, axis=0)
                kp = np.stack(kp, axis=0)
                H, inliers = cv2.findHomography(kp, kp_prev, cv2.RANSAC, 3.0)
                # If H is no good, then remove the edge.
                # TODO magic number, expose to user?
                if not find_nice_homographies(H, 1e-4) or H is None:
                    to_filter.append((node_prev, node))
                    continue
                # Otherwise
                self._registration_obj.add_edge(node_prev, node, H=H)
        # Prune the graph for all bad homographies.
        self._registration_obj.remove_edges_from(to_filter)

    @staticmethod
    def _propagate_homographies(G: nx.DiGraph) -> np.ndarray:
        # Given a directed graph of homographies, return them as a sequence of homographies for each node.
        if len(G) == 1:
            return G[list(G.nodes)[0]].get("H0", np.eye(3)[None, ...])
        elif len(G) < 1:
            raise ValueError("No nodes in graph.")
        else:
            # order the edges by dependency
            sorted_edges = list(nx.topological_sort(nx.line_graph(G)))
            sorted_H = [G[u][v]['H'] for u, v in sorted_edges]
            N0 = sorted_edges[0][0]  # first node in sorted graph
            # if node has an initial rotation
            H0 = G[N0].get("H0", np.eye(3))
            sorted_H = [H0] + sorted_H
            sorted_H = np.stack(sorted_H, axis=0)
            return np.array(tuple(
                itertools.accumulate(sorted_H, np.matmul)))  # Propagate the homographys to reference to first node.

    def _add_orientation(self):
        orientations = self._load_orientations()
        K = self._meta["intrinsics"]["K"]
        K_inv = core.inverse_K(K)
        if orientations is None:
            return
        for frame_no, data in self._registration_obj.nodes(data=True):
            pt = self._meta['time_offset'] + frame_no / self._reader_obj.fps
            # TODO: have a 0.01 s slop here, make this public to user?
            if abs(pt - orientations.times.min()) < 10e-2:
                R = orientations.rotations[0]
            elif abs(pt - orientations.times.max()) < 10e-2:
                R = orientations.rotations[-1]
            elif orientations.times.min() <= pt <= orientations.times.max():
                R = orientations(pt)
            else:
                # TODO: pt can be outside of interpolant bounds, warning the user here but could cause trouble down the
                #  pipeline.
                warnings.warn(f"playback time outside of interpolant bounds; not adding to Node.", UserWarning)
                continue
            self._registration_obj.nodes[frame_no]['H0'] = K @ R.as_matrix() @ K_inv  # Apply 3D rotation as projection homography

    def _prune_unstable_graph(self, stability_threshold: float):
        # TODO: homography can still be stable, but extremely warped. Scan through corners to find outliers (i.e. where bad warps are likely).
        """
        Stabilize the graph by pruning unstable edges iteratively.

        Args:
            stability_threshold: The threshold for determining stability.

        Returns:
            stabilized_graph: The pruned and stabilized graph.
        """
        c = 0
        flag = True
        while flag:
            flag = False  # this flag needs to be flipped to keep pruning
            c = c + 1
            # Get all the subgraphs
            subgraphs = list(nx.weakly_connected_components(self._registration_obj))
            print(f"Iteration: {c}, {len(subgraphs)} subgraphs.")
            for subgraph in (self._registration_obj.subgraph(c).copy() for c in subgraphs):
                if len(subgraph) < 2:
                    continue
                H = SequentialMosaic._propagate_homographies(subgraph)  # Get the homography sequence for the subgraph
                # Find the first unstable transformation
                is_good = find_nice_homographies(H, stability_threshold)

                # If all transformations are stable, continue on
                if all(is_good):
                    continue

                # Find the first unstable transformation's index
                unstable_index = is_good.tolist().index(False)

                # edge case: it's the first Node. Prune edge 0 -> 1
                unstable_index = unstable_index + 1 if unstable_index == 0 else unstable_index

                # Find the corresponding node in the subgraph
                nodes = list(nx.topological_sort(subgraph))

                unstable_node = nodes[unstable_index]

                # Get the predecessor of the unstable node
                predecessors = list(subgraph.predecessors(unstable_node))
                if not predecessors:
                    raise ValueError("No predecessors found, something went wrong with the graph structure.")

                # Prune the graph by removing the edge that leads to the unstable node
                pred_node = predecessors[0]
                print(f"Pruning edge: {pred_node} -> {unstable_node}")
                self._registration_obj.remove_edge(pred_node, unstable_node)
                flag = True  # flag to search for pruning again
        print(f"Done!")

    def global_registration(self):
        self._add_orientation()  # Add in extrinsic rotations as homographies to valid Nodes.
        self._prune_unstable_graph(1e-4)  # Prune the bad absolute homographies
        # Assign the absolute homography to each node in each subgraph
        for subgraph in (self._registration_obj.subgraph(c).copy() for c in
                         nx.weakly_connected_components(self._registration_obj)):
            H = self._propagate_homographies(subgraph)
            min_x, min_y, _, _ = get_mosaic_dimensions(H, self._reader_obj.width, self._reader_obj.height)
            H_t = core.homogeneous_translation(-min_x, -min_y)[None, ...]
            H = H_t @ H
            # order the edges by dependency
            sorted_nodes = list(nx. topological_sort(subgraph))
            for homography, node in zip(H, sorted_nodes):
                self._registration_obj.nodes[node]['H'] = homography

    @staticmethod
    def _bbox_overlap(shape_1: np.ndarray, shape_2: np.ndarray) -> bool:
        p1 = geometry.Polygon(shape_1)
        p2 = geometry.Polygon(shape_2)
        return p1.intersects(p2)

    def _create_tile_graph(self, tile_size: tuple[int, int]) -> nx.Graph:
        # iterate through every subgraph sequence of _registration_obj
        # get the topological sort of subgraph (a valid path)
        # get the sequence of transformations to apply to each node (include the first H0 transformation from N0).
        # Get output mosaic dimensions
        # Generate tile coordinates based on tile size parameter and output mosaic dimensions
        # For each tile
            # Create data structure for tile
            # Generate mosaic -> tile translation homography based on top left tile coordinates
            # For each node
                # warp corners to output mosaic coordinates using H0
                # determine if warped bbox overlaps with current tile coordinates
                # if overlap:
                    # H_tile = transform H0 of node with tile translation homography
                    # Add node to tile datastructure with H_tile
        """
            Create a graph where each node represents a tile of the output mosaic.
            Each tile will consist of frames whose warped coordinates overlap with the tile.

            :param tile_size: A tuple representing the width and height of each tile.
            """
        # Initialize variables
        tile_graph = nx.Graph()  # The graph where tiles will be nodes
        # Iterate over all stable subgraphs in the registration object
        for subgraph_index, subgraph in enumerate((self._registration_obj.subgraph(c).copy() for c in
                         nx.weakly_connected_components(self._registration_obj))):
            # Get the topological sort of the subgraph (a valid path of transformations)
            sorted_nodes = list(nx.topological_sort(subgraph))
            sorted_H = np.stack([subgraph.nodes[n]['H'] for n in sorted_nodes], axis=0)  # Homographies for each node

            # Get the output mosaic dimensions
            mosaic_dims = get_mosaic_dimensions(sorted_H, self._reader_obj.width, self._reader_obj.height)

            # Calculate tile coordinates
            tile_x = np.arange(0, mosaic_dims[2], tile_size[0])
            tile_y = np.arange(0, mosaic_dims[3], tile_size[1])

            for tx in tile_x:
                for ty in tile_y:
                    # Create the bounding box for the current tile
                    tile_crns = np.array([[tx, ty],
                                          [tx + tile_size[0], ty],
                                          [tx + tile_size[0], ty + tile_size[1]],
                                          [tx, ty + tile_size[1]]])

                    # Initialize a list to store frames that overlap with this tile
                    tile_frames = []

                    for frame_no, H in subgraph.nodes(data='H'):

                        # Calculate the warped corners of the frame
                        frame_crns = get_corners(H, self._reader_obj.width, self._reader_obj.height)

                        # Check if the frame_bbox overlaps with the current tile_bbox
                        if self._bbox_overlap(tile_crns, frame_crns.squeeze()):
                            # If there's an overlap, transform the homography for this tile and add to graph
                            # Get the homography
                            H_tile = core.homogeneous_translation(tx, ty) @ H
                            tile_frames.append((frame_no, H_tile))
                    if tile_frames:
                        tile_graph.add_node((subgraph_index, tx, ty), frames=tile_frames)
        return tile_graph

    def generate(self, tile_size: tuple[int, int] = (-1, -1)):
        tiles = self._create_tile_graph(tile_size)  # restructure the graph into tiles

        # Now iterate through the tiles
        for (subgraph_index, tile_x, tile_y), sequence in tiles.nodes(data='frames'):
            #  subgraph_index: which sequence this belongs to
            #  tile_x: top left corner of the tile in mosaic coordinates
            #  tile_y: top left corner of the tile in mosaic coordinates
            #  sequence: a dictionary of frame number and homography to apply to map image to the tile
            frame_numbers = [s[0] for s in sequence]
            # Get frame range
            frame_min, frame_max = min(frame_numbers), max(frame_numbers)
            self._reader_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_min)
            H = np.stack(tuple(s[1] for s in sequence), axis=0)  # TODO: sort H using frame_numbers as key
            xmin, ymin, output_width, output_height = get_mosaic_dimensions(
                H, self._reader_obj.width, self._reader_obj.height)
            # Construct mapper object
            mapper = Mapper(output_width, output_height)
            for frame_no, (h, (ret, frame)) in enumerate(zip(H, self._reader_obj)):
                frame = self._preprocessor_pipeline.apply(frame)
                mapper.update(frame, h)
                frame.release()
            output_path = self._meta["output_path"].joinpath(f"seq_{subgraph_index}_tile_{tile_x}_{tile_y}.png")
            cv2.imwrite(str(output_path), mapper.output)
            mapper.release()



def find_nice_homographies(H: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Homographies that have negative determinant will flip the coordinates. Homographies that have close to 0 will also
    have undesirable behaviour (thin lines, extreme warping etc.).
    """
    if H.ndim > 2:
        det = np.linalg.det(H[:, :2, :2])
    else:
        det = np.linalg.det(H)
    return det > eps


def get_mosaic_dimensions(H: np.ndarray, width: int, height: int) -> Sequence[int]:
    """
    Given a transformation homography, and the width and height of the input image. Calculate the bounding box of the warped image.
    """
    # Get the image corners
    dst_crn = get_corners(H, width, height)
    # Compute the top left and bottom right corners of bounding box
    return cv2.boundingRect(dst_crn.reshape(-1, 2).astype(np.float32))


def get_corners(H: np.ndarray, width: int, height: int) -> np.ndarray:
    # Get the image corners
    src_crn = np.array([[[0, 0]],
                        [[width - 1, 0]],
                        [[width - 1, height - 1, ]],
                        [[0, height - 1]]], np.float32) + 0.5
    if H.ndim > 2:
        src_crn = np.stack([src_crn] * len(H), 0)
    elif H.ndim == 2:
        src_crn = src_crn[None, ...]
        H = H[None, ...]
    src_crn_h = np.concatenate((src_crn, np.ones((len(H), 4, 1, 1))), axis=-1)
    dst_crn_h = np.swapaxes(H @ np.swapaxes(src_crn_h.squeeze(axis=2), 1, 2), 1, 2)
    dst_crn = dst_crn_h[:, :, :2] / dst_crn_h[:, :, -1:]
    return dst_crn[:, :, None, :]


def alpha_blend_cuda(img1: cv2.cuda.GpuMat, img2: cv2.cuda.GpuMat, alpha_gpu: cv2.cuda.GpuMat) -> cv2.cuda.GpuMat:
    """Perform alpha blending between two CUDA GpuMat images."""
    one_gpu = cv2.cuda.GpuMat(alpha_gpu.size(), alpha_gpu.type(), 1)
    alpha_inv_gpu = cv2.cuda.subtract(one_gpu, alpha_gpu)

    if img1.type() != alpha_gpu.type():
        img1 = cv2.cuda.merge(tuple(cv2.cuda.normalize(channel, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F, cv2.cuda.GpuMat()) for channel in cv2.cuda.split(img1)) + (alpha_gpu,), cv2.cuda.GpuMat(alpha_gpu.size(), alpha_gpu.type()))

    if img2.type() != alpha_gpu.type():
        img2 = cv2.cuda.merge(tuple(cv2.cuda.normalize(channel, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F, cv2.cuda.GpuMat()) for channel in cv2.cuda.split(img2)) + (alpha_inv_gpu, ), cv2.cuda.GpuMat(alpha_gpu.size(), alpha_gpu.type()))

    blended = cv2.cuda.alphaComp(img1, img2, cv2.cuda.ALPHA_OVER)

    return cv2.cuda.cvtColor(blended, cv2.COLOR_BGRA2BGR)
