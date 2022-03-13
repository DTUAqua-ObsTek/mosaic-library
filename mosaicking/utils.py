import argparse

import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path


def parse_args():
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
                        help="Path to .csv file containing orientation measurements that transform world to the camera frame.")
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
    parser.add_argument("-c", "--calibration", type=str, default=None,
                        help="Path to calibration file, overrides --intrinsic and --distortion.")
    parser.add_argument("-k", "--intrinsic", nargs=9, type=float, default=None,
                        help="Space delimited list of intrinsic matrix terms, Read as K[0,0],K[0,1],K[0,2],K[1,0],K[1,1],K[1,2],K[2,0],K[2,1],K[2,2]. Overriden by calibration file if intrinsic present.")
    parser.add_argument("-d", "--distortion", nargs="+", type=float, default=None,
                        help="Space delimited list of distortion coefficients, Read as K1, K2, p1, p2. Overriden by calibration file if distortion present.")
    parser.add_argument("-x", "--xrotation", type=float, default=0,
                        help="Rotation around image plane's x axis (radians).")
    parser.add_argument("-y", "--yrotation", type=float, default=0,
                        help="Rotation around image plane's y axis (radians).")
    parser.add_argument("-z", "--zrotation", type=float, default=0,
                        help="Rotation around image plane's z axis (radians).")
    parser.add_argument("-g", "--gradientclip", type=float, default=0,
                        help="Clip the gradient of severely distorted image.")
    parser.add_argument("-f", "--fisheye", action="store_true", help="Flag to use fisheye distortion model.")
    parser.add_argument("--homography", type=str, choices=["rigid", "similar", "affine", "perspective"],
                        default="similar", help="Type of 2D homography to perform.")
    group = parser.add_argument_group()
    group.add_argument("--demo", action="store_true",
                       help="Creates a video of the mosaic creation process. For demo purposes only.")
    group.add_argument("--show_demo", action="store_true", help="Display the demo while underway.")
    parser.add_argument("--features", type=str, nargs="+", choices=["ORB", "SIFT", "SURF", "BRISK", "KAZE", "ALL"],
                        default="ALL", help="Set of features to use in registration.")
    parser.add_argument("--show_matches", action="store_true", help="Display the matches.")
    parser.add_argument("--inliers_roi", action="store_true",
                        help="Only allow the convex hull of the inlier points to be displayed.")
    return parser.parse_args()


def prepare_frame(left: np.ndarray, right: np.ndarray, size: tuple):
    left_size = (int(size[0]/2), int(size[1]))
    right_size = (int(size[0]/2), int(size[1]))
    left = cv2.resize(left, left_size)
    right = cv2.resize(right, right_size)
    return np.concatenate((left, right), axis=1)


class VideoPlayer(cv2.VideoCapture):
    def __init__(self, args: argparse.Namespace):
        # if args.show:
        #     cv2.namedWindow("original", cv2.WINDOW_NORMAL)
        self._args = args
        # SETUP THE VIDEO READER
        video_path = Path(args.video).resolve()
        assert video_path.exists(), "File not found: {}".format(str(video_path))
        super().__init__(str(video_path))
        # PROPERTIES OF FRAME
        self.fps = self.get(cv2.CAP_PROP_FPS)
        self.width = int(self.get(
            cv2.CAP_PROP_FRAME_WIDTH))  # if args.scale_factor is None else int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)*args.scale_factor)
        self.height = int(self.get(
            cv2.CAP_PROP_FRAME_HEIGHT))  # if args.scale_factor is None else int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)*args.scale_factor)
        self.n_frames = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_skip = self._args.frame_skip
        self._get_starting_pos()
        print(f"Opened video {self._args.video}")

    def __iter__(self):
        return VideoPlayerIterator(self)

    def __repr__(self):
        fmt = "Frame {"+":0{}d".format(len(str(self.n_frames))) + "} "+ "of {}".format(self.n_frames)
        return fmt.format(int(self.get(cv2.CAP_PROP_POS_FRAMES)) + 1)

    def _get_starting_pos(self):
        """Set a VideoCapture object to a position (either in seconds or the frame #)"""
        if self._args.finish_time:
            self.set(cv2.CAP_PROP_POS_MSEC, self._args.finish_time * 1000.0)
            self._finish_frame = int(self.get(cv2.CAP_PROP_POS_FRAMES))
            self.set(cv2.CAP_PROP_POS_MSEC, self._args.finish_time * 1000.0)
        elif self._args.finish_frame:
            self._finish_frame = int(self._args.finish_frame - 1)
        else:
            self._finish_frame = self.n_frames - 1
        if self._args.start_time:
            self.set(cv2.CAP_PROP_POS_MSEC, self._args.start_time * 1000.0)
        elif self._args.start_frame:
            self.set(cv2.CAP_PROP_POS_FRAMES, self._args.start_frame - 1)
        else:
            self.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._start_frame = int(self.get(cv2.CAP_PROP_POS_FRAMES))
        if self._finish_frame < self._start_frame:
            raise ValueError("Finishing frame is less than starting time.")


    def evaluate_stopping(self):
        """Return true if stopping conditions met."""
        return self._finish_frame <= self.get(cv2.CAP_PROP_POS_FRAMES)

    def __del__(self):
        print(f"Closing video {self._args.video}")
        self.release()

    def __len__(self):
        return len(range(*slice(self._start_frame, self._finish_frame, self.frame_skip).indices(self.n_frames)))


class VideoPlayerIterator:
    def __init__(self, videoplayer: VideoPlayer):
        self._videoplayer = videoplayer
        self._index = self._videoplayer.get(cv2.CAP_PROP_POS_FRAMES)

    def __next__(self):
        if not self._videoplayer.evaluate_stopping():
            ret, img = self._videoplayer.read()
            if self._videoplayer.frame_skip:
                if self._index + self._videoplayer.frame_skip > self._videoplayer._finish_frame:
                    self._videoplayer.set(cv2.CAP_PROP_POS_FRAMES, self._videoplayer._finish_frame)
                else:
                    self._videoplayer.set(cv2.CAP_PROP_POS_FRAMES, self._videoplayer.get(cv2.CAP_PROP_POS_FRAMES) + self._videoplayer.frame_skip)
            if ret:
                self._index = self._videoplayer.get(cv2.CAP_PROP_POS_FRAMES)
                return img
        # End of Iteration
        raise StopIteration


def check_keypress(key: str):
    press = cv2.waitKey(1)
    if key == ord(press):
        return True
    return False


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


def load_orientations(path: os.PathLike, args):
    """Given a path containing orientations, retrieve the orientations corresponding to a time offset between video and orientation data."""
    time_offset = args.time_offset if args.time_offset else args.sync_points[1] - args.sync_points[0]
    df = pd.read_csv(str(path), index_col="timestamp")
    df.index = df.index - time_offset
    return df[~df.duplicated()]