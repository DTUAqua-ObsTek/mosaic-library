import cv2
import mosaicking.registration
from mosaicking import utils, preprocessing, registration, core, splitter
from mosaicking.core import inverse_K
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
    args = utils.parse_args()
    args.video.resolve(True)
    if not args.output_directory.exists():
        args.output_directory.mkdir()

    # Load in orientations if available
    orientation_lut = utils.load_orientation(args.orientation_file) if args.orientation_file else None
    # Create the video reader object
    reader = utils.CUDAVideoPlayer(args.video)

    # Get the frame count, image dimensions and the fps
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = reader.get(cv2.CAP_PROP_FPS)

    # Load in intrinsic data if available
    K, D = utils.parse_intrinsics(args, width, height)
    # Get the inverse intrinsic matrix for alter use
    K_inv = inverse_K(K)

    # Undistort remapper
    remappings = cv2.initUndistortRectifyMap(K, D, None, K, (width, height), cv2.CV_32FC1)
    xmap = cv2.cuda.GpuMat()
    ymap = cv2.cuda.GpuMat()
    xmap.upload(remappings[0])
    ymap.upload(remappings[1])

    # Load in detectors
    detector = registration.CompositeDetector(mosaicking.registration.parse_detectors(args.features))
    matcher = registration.CompositeMatcher()

    keypoints, descriptors = [], []

    timer = cv2.TickMeter()  # Performance Timer

    clahe = preprocessing.ColorCLAHE(40.0, (8, 8))  # Histogram equalization

    ## Part 1: Extract Features
    # If output does not have a feature .pkl file, then perform extraction
    if not args.output_directory.joinpath("features.pkl").exists():
        print(f"Performing feature extraction.")
        timer.start()
        for frame_no, (ret, frame) in enumerate(reader):
            print(f"Frame: {frame_no:05d}")
            # stop on bad frame
            if not ret:
                break
            frame = preprocessing.make_bgr(frame)  # Convert to BGR
            frame = cv2.cuda.remap(frame, xmap, ymap, cv2.INTER_CUBIC, cv2.cuda.GpuMat())  # Undistort image
            kps, descs = detector.detect(frame)  # Feature detection
            frame.release()
            keypoints.append(kps)  # Append the keypoints
            descriptors.append(descs)  # Append the descriptors
        reader = None  # Free the reader object
        keypoints = tuple(keypoints)
        descriptors = tuple(descriptors)
        with open(args.output_directory.joinpath("features.pkl"), "wb") as f:
            tmp = []
            for feature_sets in keypoints:
                features = []
                for feature_set in feature_sets:
                    features.append(cv2.KeyPoint.convert(feature_set)) # convert to numpy resource for pickling.
                tmp.append(tuple(features))
            pickle.dump((tmp, descriptors), f)
    else:
        with open(args.output_directory.joinpath("features.pkl"), "rb") as f:
            tmp, descriptors = pickle.load(f)
            keypoints = []
            for feature_sets in tmp:
                kp = []
                for feature_set in feature_sets:
                    kp.append(cv2.KeyPoint.convert(feature_set))
                keypoints.append(tuple(kp))
            keypoints = tuple(keypoints)
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
    for idx, (desc_prev, desc) in enumerate(zip(descriptors[:-1], descriptors[1:])):
        if desc is None or desc_prev is None:
            sequences.append(tuple(matches))
            continue
        all_matches = matcher.knn_match(desc, desc_prev)  # Find 2 nearest matches
        good_matches = tuple(m[0] for m in all_matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance)  # Candidate matches must pass Lowe's distance ratio test
        # Perspective Homography requires at least 4 matches, if there aren't enough matches, then we should search (maybe bag of words)
        if len(good_matches) < 10:
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
        for pair_no, (kp_prev, kp, match) in enumerate(zip(keypoints[start:stop-1], keypoints[start+1:stop], sequence)):
            kp_prev = cv2.KeyPoint.convert(tuple(chain(*kp_prev)))
            kp = cv2.KeyPoint.convert(tuple(chain(*kp)))
            idx = np.array(tuple((m.queryIdx, m.trainIdx) for m in match), dtype=int)
            src = kp[idx[:, 0], :]  # points to be transformed
            dst = kp_prev[idx[:, 1], :]  # points to align
            H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H is None:
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
            pt = args.time_offset + start / fps
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
            H = core.homogeneous_scaling(args.scale_factor)[None, ...] @ H
            absolute_sequence_graphs.append(H)
            continue
        H = np.stack((H1, ) + sequence_graph, axis=0)  # stack the entire sequence including first frame homography
        H = np.array(tuple(accumulate(H, np.matmul)))  # Propagate the homographys to make the absolute pose graph
        xmin, ymin, _, _ = get_mosaic_dimensions(H, width, height)
        t = core.homogeneous_translation(-xmin, -ymin)
        H = t[None, ...] @ H  # Apply transformation to the homographies to reference to top left corner of mosaic.
        s = core.homogeneous_scaling(args.scale_factor)
        H = s[None, ...] @ H  # Scale the homography by the scale factor
        # Calculate the dimensions of the output mosaic
        xmin, ymin, output_width, output_height = get_mosaic_dimensions(H, width, height)
        # Resize the mosaic if the mosaic memory consumption will be too high
        flag = output_width * output_height * 3 * 1 > (args.max_mem * 1.01)
        while flag:
            sf = (args.max_mem / (output_width * output_height * 3 * 1)) ** 0.5
            print(f"Calculated size {output_width * output_height * 3e-6:.1f} MB, downsizing by {sf:.5f}")
            H = core.homogeneous_scaling(sf)[None, ...] @ H
            xmin, ymin, output_width, output_height = get_mosaic_dimensions(H, width, height)
            print(f"New dims {output_width}x{output_height}\nNew MEM {output_width * output_height * 3e-6:.1f} MB")
            flag = output_width * output_height * 3 * 1 > (args.max_mem * 1.01)
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
    for sequence_tile_assignment in sequence_tile_assignments:
        num_tiles = len(sequence_tile_assignment)
        for tile in sequence_tile_assignment:
            ret = True
            c = 0
            reader = cv2.cudacodec.createVideoReader(str(args.video))
            while ret:
                ret, frame = reader.nextFrame()  # Get frame
                c = c + 1
        print("what now.")


    print(f"Building mosaics from sequences.")
    timer.start()
    reader = utils.CUDAVideoPlayer(args.video)
    # Outer loop to go over each sequence
    for seq, tile_assignment, absolute_sequence_graph in enumerate(zip(tile_assignments, absolute_sequence_graphs)):
        if absolute_sequence_graph.shape[0] < 2:
            reader.read()
            continue
        xmin, ymin, output_width, output_height = get_mosaic_dimensions(absolute_sequence_graph, width, height)
        # Begin composing the output mosaic
        output = cv2.cuda.GpuMat()
        output.upload(np.zeros((output_height, output_width, 3), dtype=np.uint8))
        output_mask = cv2.cuda.GpuMat()
        output_mask.upload(np.zeros((output_height, output_width), dtype=np.uint8))
        # The prewarped mask template
        mask = cv2.cuda.GpuMat()
        mask.upload(255 * np.ones((height, width), dtype=np.uint8))

        # Window object
        if args.show_mosaic:
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        if args.demo:
            writer = cv2.cudacodec.createVideoWriter(str(args.output_directory.joinpath(f"mosaic_{args.video.stem}_{seq:05d}{args.video.suffix}")),
                                                     (output_width, output_height), cv2.cudacodec.HEVC, fps,
                                                     cv2.cudacodec.COLOR_FORMAT_BGR)
        # Mosaic each sequence
        for idx, H in enumerate(absolute_sequence_graph):
            ret, frame = reader.nextFrame()  # Get frame
            frame = cv2.cuda.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR
            frame = cv2.cuda.remap(frame, xmap, ymap, cv2.INTER_CUBIC, cv2.cuda.GpuMat())  # Undistort
            frame = clahe.apply_color(frame)  # CLAHE on color channels.
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
            # Alpha blend the intersecting region
            blended = alpha_blend_cuda(output, warped, alpha_gpu)
            # Convert to 8UC3
            blended = cv2.cuda.merge(tuple(
               cv2.cuda.normalize(channel, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_8U, cv2.cuda.GpuMat()) for channel in
               cv2.cuda.split(blended)), cv2.cuda.GpuMat())
            blended.copyTo(mask_intersect, output)
            if args.demo:
                writer.write(output)
            if args.show_mosaic:
                cv2.imshow("output", output.download())
                cv2.waitKey(30)
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

        if args.show_mosaic:
            cv2.destroyWindow("output")
        if args.demo:
            writer.release()
        write_success = cv2.imwrite(str(args.output_directory.joinpath(f"mosaic_{args.video.stem}_{seq:05d}.png")), output.download())
        output.release()
        output_mask.release()
        mask.release()

    timer.stop()
    print(f"Mosaicking took {timer.getTimeSec():.2f} seconds.")
    timer.reset()


if __name__ == "__main__":
    main()
