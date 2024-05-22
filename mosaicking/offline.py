import cv2
import mosaicking
from mosaicking.utils import parse_args, VideoPlayer, parse_intrinsics
from registration import OrbDetector, Matcher, SiftDetector
import numpy as np
from itertools import accumulate
import math
from scipy.spatial.transform import Slerp, Rotation
import pandas as pd
from pathlib import Path
from typing import Sequence
import pickle


def nice_homography(H: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    """

    if H.ndim > 2:
        det = np.linalg.det(H[:, :2, :2])
    else:
        det = np.linalg.det(H)
    return det > eps


def get_mosaic_dimensions(H: np.ndarray, width: int, height: int) -> Sequence[int]:
    # Get the image corners
    src_crn = np.array([[[0, 0]],
                        [[width - 1, 0]],
                        [[width - 1, height - 1, ]],
                        [[0, height - 1]]], np.float32) + 0.5
    # convert to homogeneous representation (x, y, w)
    src_crn_h = cv2.convertPointsToHomogeneous(src_crn)
    # Broadcast matrix multiplication of homographys with homogeneous coordinate corners
    src_crn_h = src_crn_h.reshape(-1, 1, 3)
    dst_crn_h = np.swapaxes(H @ src_crn_h.squeeze().T, 1, 2)
    dst_crn = cv2.convertPointsFromHomogeneous(dst_crn_h.reshape(-1, 1, 3)).reshape(H.shape[0], 4, 1, 2)
    # Compute the top left and bottom right corners of bounding box
    return cv2.boundingRect(dst_crn.reshape(-1, 2).astype(np.float32))


def generate_plane(bbox: tuple[float, float, float, float], n_points: int, offset: float = 1) -> tuple[np.ndarray, ...]:
    x_min, y_min, w, h = bbox  # extract the top left corodinates and width / height of the box
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
    rays = np.ones((uv.shape[0], 3))
    rays[:, :2] = (uv - K[:2, 2]) / K[[0, 1], [0, 1]]
    return rays


def ray_to_pixel(ray: np.ndarray, K: np.ndarray) -> np.ndarray:
    return K @ ray.T


def load_orientation(orientation_file: Path, video_time_offset: float = 0.0) -> Slerp:
    df = pd.read_csv(orientation_file)
    df["pt"] = df["ts"] - video_time_offset
    return Slerp(df['ts'], Rotation.from_quat(df.loc[:, ['qx', 'qy', 'qz', 'qw']]))


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


def make_3_channel_mask(mask_gpu: cv2.cuda.GpuMat) -> cv2.cuda.GpuMat:
    """Convert a single-channel GPU mask to a 3-channel GPU mask using CUDA merge."""
    channels = [mask_gpu, mask_gpu, mask_gpu]
    mask_3_channel_gpu = cv2.cuda.merge(channels, cv2.cuda.GpuMat())
    return mask_3_channel_gpu


def mask_color_image(image: cv2.cuda.GpuMat, mask: cv2.cuda.GpuMat) -> cv2.cuda.GpuMat:
    channels = cv2.cuda.split(image)
    return cv2.cuda.merge(tuple(cv2.cuda.bitwise_and(channel, channel, mask=mask) for channel in channels), cv2.cuda.GpuMat())


def inverse_K(K: np.ndarray) -> np.ndarray:
    K_inv = np.eye(*K.shape)
    K_inv[0, 0] = K[1, 1]
    K_inv[1, 1] = K[0, 0]
    K_inv[0, 1] = -K[0, 1]
    K_inv[0, 2] = K[1, 2] * K[0, 1] - K[0, 2] * K[1, 1]
    K_inv[1, 2] = -K[1, 2] * K[0, 0]
    K_inv[2, 2] = K[0, 0] * K[1, 1]
    return 1 / (K[0, 0] * K[1, 1]) * K_inv


def remove_z_rotation(rotation: Rotation) -> Rotation:
    """
    Removes the extrinsic Z rotation (yaw) component from a Rotation object.

    Parameters:
    rotation (Rotation): A scipy Rotation object.

    Returns:
    Rotation: A new Rotation object with the Z rotation component removed.
    """
    # Decompose the original rotation into Euler angles (yaw, pitch, roll)
    # 'zyx' means the input rotation is first around z, then y, then x
    ypr = rotation.as_euler('zyx')

    # Set yaw to zero to remove rotation about the Z-axis
    if ypr.ndim > 1:
        ypr[:, 0] = 0
    else:
        ypr[0] = 0

    # Create a new rotation using the modified yaw and original pitch and roll
    # Note that the angles must be provided in the reverse order of axes
    new_rotation = Rotation.from_euler('zyx', ypr)

    return new_rotation


def main():
    args = parse_args()
    args.video.resolve(True)

    orientation_lut = load_orientation(args.orientation_file) if args.orientation_file else None

    reader = cv2.cudacodec.createVideoReader(str(args.video))  # VideoReader object

    _, frame_count = reader.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = int(frame_count)
    _, width = reader.get(cv2.CAP_PROP_FRAME_WIDTH)
    width = int(width)
    _, height = reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height = int(height)
    _, fps = reader.get(cv2.CAP_PROP_FPS)

    K, D = parse_intrinsics(args, width, height)
    K_inv = np.linalg.inv(K)

    remappings = cv2.initUndistortRectifyMap(K, D, None, K, (width, height), cv2.CV_32FC1)
    xmap = cv2.cuda.GpuMat()
    ymap = cv2.cuda.GpuMat()
    xmap.upload(remappings[0])
    ymap.upload(remappings[1])
    keypoints, descriptors = [], []

    detector = OrbDetector()
    #detector = SiftDetector()
    matcher = Matcher()

    timer = cv2.TickMeter()

    ## Part 1: Extract Features
    if False:
        timer.start()
        for frame_no in range(frame_count):
            print(f"Frame: {frame_no:05d}")
            ret, frame = reader.nextFrame()
            if not ret:
                break
            frame = cv2.cuda.remap(frame, xmap, ymap, cv2.INTER_CUBIC, cv2.cuda.GpuMat())
            kps, descs = detector.detect(frame)
            keypoints.append(kps)
            descriptors.append(descs)
        reader = None
        keypoints = tuple(keypoints)
        descriptors = tuple(descriptors)

        timer.stop()
        print(f"Feature Extraction took {timer.getTimeSec():.2f} seconds.")
        timer.reset()
        if False:
            with open("features.pkl", "wb") as f:
                tmp = tuple(cv2.KeyPoint.convert(k) for k in keypoints)
                pickle.dump((tmp, descriptors), f)
    else:
        with open("features.pkl", "rb") as f:
            tmp, descriptors = pickle.load(f)
            keypoints = tuple(cv2.KeyPoint.convert(k) for k in tmp)

    ## Part 2: Feature Matching
    # For now, this just splits the video into sequences of consecutive matches.
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

    ## Part 3: Homography Computation
    timer.start()
    sequence_graphs = []
    for seq, ((start, stop), sequence) in enumerate(zip(sequence_idx, sequences)):
        homography_graph = []
        for pair_no, (kp_prev, kp, match) in enumerate(zip(keypoints[start:stop-1], keypoints[start+1:stop], sequence)):
            kp_prev = cv2.KeyPoint.convert(kp_prev)
            kp = cv2.KeyPoint.convert(kp)
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
    timer.start()

    ## Part 4: Mosaic generation
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
            H1 = K @ R.as_matrix() @ K_inv
        else:
            H1 = np.eye(3)
        if not sequence_graph:
            H = np.stack((H1, ), axis=0)
            H[:, :2, :] = args.scale_factor * H[:, :2, :]
            absolute_sequence_graphs.append(H)
            continue
        H = np.stack((H1, ) + sequence_graph, axis=0)  # stack the entire sequence including first frame homography
        H = np.array(tuple(accumulate(H, np.matmul)))  # Propagate the homographys to make the absolute pose graph
        xmin, ymin, _, _ = get_mosaic_dimensions(H, width, height)
        H = np.array([[[1.0, 0.0, -xmin],
                       [0.0, 1.0, -ymin],
                       [0.0, 0.0, 1.0]],]) @ H
        H[:, :2, :] = args.scale_factor * H[:, :2, :]  # Scale the homography by the scale factor
        absolute_sequence_graphs.append(H)  # Append to the list
    absolute_sequence_graphs = tuple(absolute_sequence_graphs)  # immutable

    # Check homographys for unwanted behaviours
    for H in absolute_sequence_graphs:
        # Calculate the dimensions of the output mosaic
        xmin, ymin, output_width, output_height = get_mosaic_dimensions(H, width, height)
        # Resize the mosaic if the mosaic memory consumption will be too high
        flag = output_width * output_height * 24 > 3.2e9
        while flag:
            sf = 3.2e9 / (output_width * output_height * 24)
            print(f"Calculated size {output_width * output_height * 24e-9:.2f} GB, recommend downsize by {sf:.5f}")
            H[:, :2, :] = sf * H[:, :2, :]
            xmin, ymin, output_width, output_height = get_mosaic_dimensions(H, width, height)
            print(f"New dims {output_width}x{output_height}\nNew MEM {output_width * output_height * 24e-9:.2f} GB")
            flag = output_width * output_height * 24 > 3.2e9
        # Determine if orientation is preserved
        #idx = nice_homography(H, 5e-4)

    # Filter homographys for unwanted behaviours

    # Build the resulting mosaic
    reader = cv2.cudacodec.createVideoReader(str(args.video))
    # Outer loop to go over each sequence
    for absolute_sequence_graph in absolute_sequence_graphs:
        if absolute_sequence_graph.shape[0] < 2:
            reader.nextFrame()
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
            writer = cv2.cudacodec.createVideoWriter(str(args.video.with_name("mosaic_"+args.video.name)),
                                                     (output_width, output_height), cv2.cudacodec.HEVC, fps,
                                                     cv2.cudacodec.COLOR_FORMAT_BGR)
        # Go through each sequence
        for idx, H in enumerate(absolute_sequence_graph):
            ret, frame = reader.nextFrame()  # Get frame

            frame = cv2.cuda.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR
            frame = cv2.cuda.remap(frame, xmap, ymap, cv2.INTER_CUBIC, cv2.cuda.GpuMat())
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
            # blended.copyTo(mask_intersect, output)
            if args.demo:
                writer.write(output)
            if args.show_mosaic:
                cv2.imshow("output", output.download())
                cv2.waitKey(30)

        if args.show_mosaic:
            cv2.destroyWindow("output")
        if args.demo:
            writer.release()
        break
    timer.stop()
    print(f"Mosaicking took {timer.getTimeSec():.2f} seconds.")
    timer.reset()



if __name__ == "__main__":
    main()
