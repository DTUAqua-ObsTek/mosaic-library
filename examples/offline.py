import cv2
import mosaicking.registration
import mosaicking.transformations
from mosaicking import utils, preprocessing, registration, splitter
from mosaicking.transformations import inverse_K
import numpy as np
from itertools import accumulate, chain
from scipy.spatial.transform import Rotation
import pickle
from mosaicking.mosaic import get_mosaic_dimensions, alpha_blend_cuda


def generate_grid(bbox: tuple[float, float, float, float], n_points: int, offset: float = 1) -> tuple[np.ndarray, ...]:
    """
    Create a 2D homogeneous coordinate grid defined by a bounding box and points.
    """
    x_min, y_min, w, h = bbox  # extract the top left coordinates and width / height of the box
    x_max = x_min + w
    y_max = y_min + h
    x_anchors = np.linspace(x_min, x_max, n_points)
    y_anchors = np.linspace(y_min, y_max, n_points)
    vert_lines = []
    horz_lines = []
    for x in x_anchors:
        vert_lines.append(np.array([[[x, y_min, offset]],
                                    [[x, y_max, offset]]]))
    for y in y_anchors:
        horz_lines.append(np.array([[[x_min, y, offset]],
                                    [[x_max, y, offset]]]))

    return tuple(vert_lines + horz_lines)


def project_world_lines(lines: tuple[np.ndarray, ...], R: Rotation, img: np.ndarray, K: np.ndarray, D: np.ndarray = None, t: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    lines: arrays of 3D points in world coordinates
    R: Rotation object describing camera -> world
    T: Translation describing camera -> world
    img: input image
    """
    image_lines = [cv2.projectPoints(line, R.inv().as_rotvec(), -R.inv().apply(t), K, D)[0].astype(int) for line in lines]
    draw_lines = []
    for image_line in image_lines:
        ret, p1, p2 = cv2.clipLine((0, 0, ) + img.shape[1::-1], image_line[0, 0], image_line[1, 0])
        if ret:
            draw_lines.append([p1, p2])

    return cv2.polylines(img, image_lines, False, (0, 255, 0), 1, cv2.LINE_AA)


def pixel_to_3D_ray(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Trace 3D ray vector from a given pixel and camera matrix."""
    rays = np.ones((uv.shape[0], 3))
    rays[:, :2] = (uv - K[:2, 2]) / K[[0, 1], [0, 1]]
    return rays


def ray_to_pixel(ray: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project a ray into pixel coordinates"""
    return K @ ray.T


def make_3_channel_mask(mask_gpu: cv2.cuda.GpuMat) -> cv2.cuda.GpuMat:
    """Convert a single-channel GPU mask to a 3-channel GPU mask using CUDA merge."""
    channels = [mask_gpu, mask_gpu, mask_gpu]
    mask_3_channel_gpu = cv2.cuda.merge(channels, cv2.cuda.GpuMat())
    return mask_3_channel_gpu


def mask_color_image(image: cv2.cuda.GpuMat, mask: cv2.cuda.GpuMat) -> cv2.cuda.GpuMat:
    """Preserve masked regions of a multi-channel image using CUDA merge."""
    channels = cv2.cuda.split(image)
    return cv2.cuda.merge(tuple(cv2.cuda.bitwise_and(channel, channel, mask=mask) for channel in channels), cv2.cuda.GpuMat())


def main():
    max_mem = 4e9  # Max allowed memory 4 GB

    args = utils.parse_args()
    args.video.resolve(True)
    args.project.mkdir(exist_ok=True, parents=True)

    # Load in orientations if available
    orientation_lut = utils.load_orientation_slerp(args.orientation_path, args.orientation_time_offset) if args.orientation_path and args.orientation_time_offset else None
    # Create the video reader object
    reader = utils.CUDAVideoPlayer(args.video)

    # Get the frame count, image dimensions and the fps
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = reader.get(cv2.CAP_PROP_FPS)

    # Load in intrinsic data if available
    K = np.eye(3)
    K[[0, 1], [0, 1]] = width * height
    K[[0, 1], [2, 2]] = [width / 2, height / 2]
    D = np.zeros(5)
    if args.calibration:
        calibration = utils.parse_intrinsics(args.calibration)
        K = calibration.get("K")
        D = calibration.get("D")
        # Get the inverse intrinsic matrix for alter use
    K_inv = inverse_K(K)

    # Undistort remapper
    remappings = cv2.initUndistortRectifyMap(K, D, None, K, (width, height), cv2.CV_32FC1)
    xmap = cv2.cuda.GpuMat()
    ymap = cv2.cuda.GpuMat()
    xmap.upload(remappings[0])
    ymap.upload(remappings[1])

    # Load in detectors
    detector = registration.CompositeDetector(args.feature_types, nfeatures=4000)
    matcher = registration.CompositeMatcher()

    features = []

    timer = cv2.TickMeter()  # Performance Timer

    clahe = preprocessing.ColorCLAHE(40.0, (8, 8))  # Histogram equalization

    ## Part 1: Extract Features
    # If output does not have a feature .pkl file, then perform extraction
    if not args.project.joinpath("features.pkl").exists():
        print(f"Performing feature extraction.")
        timer.start()
        for ret, frame_no, name, frame in reader:
            print(f"Frame: {frame_no:05d}")
            # stop on bad frame
            if not ret:
                break
            frame = preprocessing.make_bgr(frame)  # Convert to BGR
            frame = cv2.cuda.remap(frame, xmap, ymap, cv2.INTER_CUBIC, cv2.cuda.GpuMat())  # Undistort image
            features.append(detector.detect(frame))  # Feature extraction
            frame.release()
        reader = None  # Free the reader object
        with open(args.project.joinpath("features.pkl"), "wb") as f:
            for i, feature_set in enumerate(features):
                for feature_type in args.feature_types:
                    feature_set[feature_type]["keypoints"] = cv2.KeyPoint.convert(feature_set[feature_type]["keypoints"])
                features[i] = feature_set
            pickle.dump(features, f)
    else:
        with open(args.project.joinpath("features.pkl"), "rb") as f:
            features = pickle.load(f)
            for i, feature_set in enumerate(features):
                for feature_type in args.feature_types:
                    feature_set[feature_type]["keypoints"] = cv2.KeyPoint.convert(feature_set[feature_type]["keypoints"])
                features[i] = feature_set
    timer.stop()
    print(f"Feature Extraction took {timer.getTimeSec():.2f} seconds.")
    timer.reset()

    ## Part 2: Feature Matching
    # For now, this just splits the video into sequences of consecutive matches.
    print(f"Performing feature matching.")
    timer.start()
    sequences = []
    matches = []
    # Go through each image pair sequentially
    for idx, (feature_set_prev, feature_set) in enumerate(zip(features[:-1], features[1:])):
        if not feature_set_prev or not feature_set:
            sequences.append(tuple(matches))
            continue
        descriptors_prev = {feature_type: extracted_features['descriptors'] for feature_type, extracted_features in
                            feature_set_prev.items()}
        descriptors = {feature_type: extracted_features['descriptors'] for feature_type, extracted_features in
                       feature_set.items()}
        all_matches = matcher.knn_match(descriptors, descriptors_prev)  # Find 2 nearest matches
        good_matches = dict()
        # Apply Lowe's distance ratio test to acquire good matches.
        for feature_type, found_matches in all_matches.items():
            good_matches.update(
                {feature_type: tuple(m[0] for m in found_matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance)})
        num_matches = sum([len(m) for m in good_matches.values()])
        # Perspective Homography requires at least 4 matches, if there aren't enough matches, then we should search (maybe bag of words)
        if num_matches < args.min_matches:
            sequences.append(tuple(matches))
            matches = []
            continue
        matches.append(good_matches)
    sequences.append(tuple(matches))
    sequences = tuple(sequences)

    # Get the frame indices of all the sequences
    sequence_idx = []
    frame_no = -1
    for seq, sequence in enumerate(sequences):
        frame_no += 1
        sequence_idx.append((frame_no, frame_no + len(sequence)))
        frame_no += len(sequence)

    timer.stop()
    print(f"Brute Force Matching took {timer.getTimeSec():2f} seconds.")
    timer.reset()

    # ## Part 2.5: FLANN sequence matching.
    # # Here I want to take the sequences and see if I can match features from new sequences with features from old sequences.
    # # Then I should be able to merge matched sequences together, reducing number of mosaics.
    #
    # # FLANN parameters
    # index_params = dict(algorithm=255)  # algorithm 255 for autotuning
    # search_params = dict(checks=5)  # specify the number of checks
    #
    # # Create a FLANN Matcher object
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    #
    # # Take the upper triangle of combinations
    # for seq1, ((start1, stop1), sequence1) in enumerate(zip(sequence_idx[:-1], sequences[:-1])):
    #     # TODO: this type of concatenation doesn't work for multiple features
    #     desc1 = np.concatenate(descriptors[start1: stop1], axis=1)  # combine the from the sequence into 1 for searching
    #     desc1_idx = []
    #     for desc_idx, d in enumerate(sequence1):
    #         desc1_idx.append([desc])
    #
    #     desc1_idx = {desc_idx: image_idx }
    #     for seq2, ((start2, stop2), sequence2) in enumerate(zip(sequence_idx[seq1 + 1:], sequences[seq1 + 1:]), start=seq1 + 1):
    #         # TODO: same story here
    #         desc2 = np.concatenate(descriptors[start2: stop2], axis=1)
    #         matches = flann.knnMatch(desc1.squeeze().astype(np.float32), desc2.squeeze().astype(np.float32), k=2)
    #         good_matches = tuple(m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance)  # Candidate matches must pass Lowe's distance ratio test

    ## Part 3: Homography Computation
    print(f"Performing homography estimation.")
    timer.start()
    sequence_graphs = []
    for seq, ((start, stop), sequence) in enumerate(zip(sequence_idx, sequences)):
        homography_graph = []
        for pair_no, (feature_set_prev, feature_set, match) in enumerate(zip(features[start:stop-1], features[start+1:stop], sequence)):
            kp = tuple(feature_set[feature_type]['keypoints'] for feature_type in args.feature_types)
            kp_prev = tuple(feature_set_prev[feature_type]['keypoints'] for feature_type in args.feature_types)
            kp = tuple(chain(*kp))
            kp_prev = tuple(chain(*kp_prev))
            kp = np.array(kp) if not isinstance(kp[0], cv2.KeyPoint) else cv2.KeyPoint.convert(kp)
            kp_prev = np.array(kp_prev) if not isinstance(kp_prev[0], cv2.KeyPoint) else cv2.KeyPoint.convert(kp_prev)
            idx = np.array(tuple((m.queryIdx, m.trainIdx) for m in match[feature_type] for feature_type in args.feature_types))
            src = kp[idx[:, 0], :]  # points to be transformed
            dst = kp_prev[idx[:, 1], :]  # points to align
            # align points in current frame to points in previous frame
            H, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.0)
            if H is not None:
                H = np.concatenate((H, ((0., 0., 1.),)), axis=0)
            if H is None or not mosaicking.mosaic.find_nice_homographies(H, args.epsilon):
                sequence_graphs.append(tuple(homography_graph))
                continue
            homography_graph.append(H)
        sequence_graphs.append(tuple(homography_graph))
    sequence_graphs = tuple(sequence_graphs)

    timer.stop()
    print(f"Homography estimation took {timer.getTimeSec():.2f} seconds.")
    timer.reset()

    ## Part 4: Obtain and refine mosaic properties
    print(f"Calculating mosaic requirements.")
    timer.start()
    # Build absolute pose graphs (referenced to the start of each sequence)
    absolute_sequence_graphs = []
    for seq, ((start, stop), sequence_graph) in enumerate(zip(sequence_idx, sequence_graphs)):
        # If orientation is provided, then apply it to the reference image (otherwise just I)
        if orientation_lut:
            pt = args.orientation_time_offset + start / fps
            if abs(pt - orientation_lut.times.min()) < 10e-3:
                R = orientation_lut.rotations[0]
            else:
                R = orientation_lut(pt)
            H1 = K @ R.as_matrix() @ K_inv  # Apply 3D rotation
        else:
            H1 = np.eye(3)  # No rotation to apply
        # If the sequence graph is empty, then only apply the initial transformation
        if not sequence_graph:
            H = np.stack((H1, ), axis=0)
            # Disabled resizing for now
            # H = mosaicking.transformations.homogeneous_scaling(args.scale_factor)[None, ...] @ H
            absolute_sequence_graphs.append(H)
            continue
        H = np.stack((H1, ) + sequence_graph, axis=0)  # stack the entire sequence including first frame homography
        H = np.array(tuple(accumulate(H, np.matmul)))  # Propagate the homographys to make the absolute pose graph
        xmin, ymin, _, _ = get_mosaic_dimensions(H, width, height)
        t = mosaicking.transformations.homogeneous_translation(-xmin, -ymin)
        H = t[None, ...] @ H  # Apply transformation to the homographies to reference to top left corner of mosaic.
        #s = mosaicking.transformations.homogeneous_scaling(args.scale_factor)
        #H = s[None, ...] @ H  # Scale the homography by the scale factor
        # Calculate the dimensions of the output mosaic
        xmin, ymin, output_width, output_height = get_mosaic_dimensions(H, width, height)
        # Resize the mosaic if the mosaic memory consumption will be too high
        flag = output_width * output_height * 3 * 1 > (max_mem * 1.01)
        while flag:
            sf = (max_mem / (output_width * output_height * 3 * 1)) ** 0.5
            print(f"Calculated size {output_width * output_height * 3e-6:.1f} MB, downsizing by {sf:.5f}")
            H = mosaicking.transformations.homogeneous_scaling(sf)[None, ...] @ H
            xmin, ymin, output_width, output_height = get_mosaic_dimensions(H, width, height)
            print(f"New dims {output_width}x{output_height}\nNew MEM {output_width * output_height * 3e-6:.1f} MB")
            flag = output_width * output_height * 3 * 1 > (max_mem * 1.01)
        absolute_sequence_graphs.append(H)  # Append to the list
    absolute_sequence_graphs = tuple(absolute_sequence_graphs)  # immutable

    timer.stop()
    print(f"Mosaic allocation took {timer.getTimeSec():.2f} seconds.")
    timer.reset()

    # Part 4.5: Allocate images to tiles.
    # For each sequence, we specify in tile major order: the frame sequence and corresponding homography.
    sequence_tile_assignments = []
    for (start, stop), absolute_sequence_graph in zip(sequence_idx, absolute_sequence_graphs):
        tile_assignments = []
        # Bounds of the mosaic
        xmin, ymin, output_width, output_height = get_mosaic_dimensions(absolute_sequence_graph, width, height)
        # The size of the tiles
        tile_size = args.tile_size if args.tile_size > 0 else max(output_width, output_height)
        # The corners of all the tiles, referenced to the top left corner of the mosaic.
        tile_corners = splitter.calculate_tiles((xmin, ymin, output_width, output_height), (tile_size, tile_size))
        for tile_idx, (tile_xmin, tile_ymin, tile_xmax, tile_ymax) in enumerate(tile_corners):
            tile_assignment = []
            for seq_idx, H in enumerate(absolute_sequence_graph):
                src_crn = np.array([[[0, 0]],
                                    [[width - 1, 0]],
                                    [[width - 1, height - 1, ]],
                                    [[0, height - 1]]], np.float32) + 0.5
                # convert to homogeneous representation (x, y, w)
                src_crn_h = cv2.convertPointsToHomogeneous(src_crn)
                # Broadcast matrix multiplication of homographys with homogeneous coordinate corners
                src_crn_h = src_crn_h.reshape(-1, 1, 3)
                dst_crn_h = (H @ src_crn_h.squeeze().T).T
                dst_crn = cv2.convertPointsFromHomogeneous(dst_crn_h.reshape(-1, 1, 3)).squeeze()
                if np.any((dst_crn[:, 0] >= tile_xmin) & (dst_crn[:, 0] < tile_xmax) &
                          (dst_crn[:, 1] >= tile_ymin) & (dst_crn[:, 1] < tile_ymax)):
                    H_new = splitter.calculate_translation_homography(H, (tile_xmin, tile_ymin))
                    tile_assignment.append((seq_idx + start, H_new))
            tile_assignments.append(tuple(tile_assignment))
        sequence_tile_assignments.append(tuple(tile_assignments))


    # Part 5: Build each tile
    print(f"Building mosaics from sequences.")
    timer.start()
    reader = utils.CUDAVideoPlayer(args.video)
    # Outer loop to go over each sequence
    for seq, (tile_assignment, absolute_sequence_graph) in enumerate(zip(sequence_tile_assignments, absolute_sequence_graphs)):
        if absolute_sequence_graph.shape[0] < 2:
            reader.read()
            continue
        xmin, ymin, output_width, output_height = get_mosaic_dimensions(absolute_sequence_graph, width, height)
        output_height, output_width = np.ceil(output_height).astype(int), np.ceil(output_width).astype(int)
        # Begin composing the output mosaic
        output = cv2.cuda.GpuMat()
        output.upload(np.zeros((output_height, output_width, 3), dtype=np.uint8))
        output_mask = cv2.cuda.GpuMat()
        output_mask.upload(np.zeros((output_height, output_width), dtype=np.uint8))
        # The prewarped mask template
        mask = cv2.cuda.GpuMat()
        mask.upload(255 * np.ones((height, width), dtype=np.uint8))

        # Mosaic each sequence
        for idx, H in enumerate(absolute_sequence_graph):
            ret, frame_no, name, frame = reader.read()  # Get frame
            frame = cv2.cuda.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR
            frame = cv2.cuda.remap(frame, xmap, ymap, cv2.INTER_CUBIC, cv2.cuda.GpuMat())  # Undistort
            frame = clahe.apply(frame)  # CLAHE on color channels.
            warped = cv2.cuda.warpPerspective(frame, H, (output_width, output_height), None, cv2.INTER_CUBIC,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            warped_mask = cv2.cuda.warpPerspective(mask, H,
                                                   (output_width, output_height), None, cv2.INTER_CUBIC,
                                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            # Create binary masks for the regions
            output_mask_bin = cv2.cuda.threshold(output_mask, 1, 255, cv2.THRESH_BINARY)[1]
            warped_mask_bin = cv2.cuda.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)[1]

            # Identify the intersecting and exclusive regions
            mask_intersect = cv2.cuda.bitwise_and(output_mask_bin, warped_mask_bin)
            output_mask_only = cv2.cuda.bitwise_and(output_mask_bin, cv2.cuda.bitwise_not(warped_mask_bin))
            warped_mask_only = cv2.cuda.bitwise_and(warped_mask_bin, cv2.cuda.bitwise_not(output_mask_bin))

            # Copy the warped region to the exclusively warped region (that's it for now)
            warped.copyTo(warped_mask_only, output)
            # Update the output mask with the warped region mask
            warped_mask_only.copyTo(warped_mask_only, output_mask)

            # Blend the intersecting regions
            # Prepare an alpha blending mask
            alpha_gpu = cv2.cuda.normalize(mask_intersect, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F, cv2.cuda.GpuMat())
            alpha_gpu = alpha_gpu.convertTo(alpha_gpu.type(), alpha=args.alpha)
            # Alpha blend the intersecting region
            blended = alpha_blend_cuda(output, warped, alpha_gpu)
            # Convert to 8UC3
            blended = cv2.cuda.merge(tuple(
               cv2.cuda.normalize(channel, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_8U, cv2.cuda.GpuMat()) for channel in
               cv2.cuda.split(blended)), cv2.cuda.GpuMat())
            blended.copyTo(mask_intersect, output)
            frame.release()
            warped.release()
            warped_mask.release()
            output_mask_bin.release()
            warped_mask_bin.release()
            mask_intersect.release()
            output_mask_only.release()
            warped_mask_only.release()
            alpha_gpu.release()
            blended.release()

        write_success = cv2.imwrite(str(args.project.joinpath(f"mosaic_{args.video.stem}_{seq:05d}.png")), output.download())
        output.release()
        output_mask.release()
        mask.release()

    timer.stop()
    print(f"Mosaicking took {timer.getTimeSec():.2f} seconds.")
    timer.reset()


if __name__ == "__main__":
    main()
