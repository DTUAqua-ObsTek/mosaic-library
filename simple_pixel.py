import cv2
import argparse
from pathlib import Path
import sys
import numpy as np


def get_features(img: np.ndarray, fdet: cv2.Feature2D, mask=None):
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return fdet.detectAndCompute(img, mask)


def get_starting_pos(cap: cv2.VideoCapture, args):
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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to video file.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start_time", type=float, help="Time (secs) to start from.")
    group.add_argument("--start_frame", type=int, help="Frame number to start from.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--finish_time", type=float, help="Time (secs) to finish at.")
    group.add_argument("--finish_frame", type=int, help="Frame number to finish at.")
    parser.add_argument("--frame_skip", type=int, default=None, help="Number of frames to skip between each mosaic update.")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    assert video_path.exists(), "File not found: {}".format(str(video_path))

    reader = cv2.VideoCapture(str(video_path))
    reader = get_starting_pos(reader, args)
    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    formatspec = "{:0"+"{}d".format(len(str(n_frames)))+"}"
    cv2.namedWindow(str(video_path), cv2.WINDOW_AUTOSIZE)

    mapper = cv2.reg_MapperGradProj()

    H_mos = None

    first = True
    while not evaluate_stopping(reader, args):
        if args.frame_skip is not None and not first:
            reader.set(cv2.CAP_PROP_POS_FRAMES, reader.get(cv2.CAP_PROP_POS_FRAMES)+args.frame_skip-1)
        # Acquire a frame
        ret, img = reader.read()
        if not ret:
            sys.stderr.write("Frame missing: {}\n".format(formatspec.format(int(reader.get(cv2.CAP_PROP_POS_FRAMES)))))
            continue
        elif first:
            mosaic_img = img.copy()
            prev_img = img.copy()
            first = False
            continue

        # Preprocess the image

        # Map the current image onto the previous image
        mapping = mapper.calculate(prev_img, img)

        out = mapping.warp(img)

        # Initialise the mosaic homography with the first homography
        if H_mos is None:
            H_mos = H.copy()
        # Otherwise, propagate the homography transform to the root image
        else:
            H_mos = H_mos @ H

        # Get the corners of the current image in homogeneous coords (X,Y,Z=0,W=1)
        src_crn = np.array([[0, width, width, 0],
                            [0, 0, height, height],
                            [1,1,1,1]], np.float)
        # Get the corners of the mosaic image in homogeneous coords (X,Y,Z=0,W=1)
        dst_crn = np.array([[0, mosaic_img.shape[1], mosaic_img.shape[1], 0],
                            [0, 0, mosaic_img.shape[0], mosaic_img.shape[0]],
                            [1, 1, 1, 1]], np.float)

        # Warp the src corners to get them into the same plane as dst
        warp_dst = H_mos @ src_crn

        # Concatenate
        pts = np.concatenate([dst_crn, warp_dst], axis=1)
        # Convert to cartesian coords
        pts = pts[:2,:]/pts[-1,:]

        xmin,ymin = np.int32(pts.min(axis=1) - 0.5)
        xmax,ymax = np.int32(pts.max(axis=1) + 0.5)

        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate
        result = cv2.warpPerspective(img, Ht.dot(H_mos), (xmax - xmin, ymax - ymin))
        result[t[1]:(mosaic_img.shape[0] + t[1]), t[0]:(mosaic_img.shape[1] + t[0])] = mosaic_img
        cv2.imshow(str(video_path), result)

        # Update the previous frame
        prev_img = img.copy()
        kp_prev = kp
        des_prev = des.copy()
        mosaic_img = result.copy()

        # After image processing is done
        img = cv2.putText(img, "{:.2f}".format(reader.get(cv2.CAP_PROP_POS_MSEC)/1000.0),
                         (0, height-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        img = cv2.putText(img, "{}".format(formatspec.format(int(reader.get(cv2.CAP_PROP_POS_FRAMES)+1))),
                         (0, height-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        # cv2.imshow(str(video_path), img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            sys.stdout.write("Quitting.\n")
            break
    cv2.destroyWindow(str(video_path))
    reader.release()