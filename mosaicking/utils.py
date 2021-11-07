import cv2
import os
import pandas as pd
import numpy as np


def prepare_frame(left: np.ndarray, right: np.ndarray, size: tuple):
    left_size = (int(size[0]/2), int(size[1]))
    right_size = (int(size[0]/2), int(size[1]))
    left = cv2.resize(left, left_size)
    right = cv2.resize(right, right_size)
    return np.concatenate((left, right), axis=1)


class VideoPlayer:
    def __init__(self, file: str, show: bool = False):
        if show:
            cv2.namedWindow("original", cv2.WINDOW_NORMAL)


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