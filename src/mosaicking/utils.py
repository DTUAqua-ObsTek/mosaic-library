import argparse
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Union, Type, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from scipy.spatial.transform import Slerp, Rotation

import cv2

import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


# Define a TypeVar bound to the abstract base class
T = TypeVar('T', bound='DataReader')


def parse_intrinsics(calibration_file: Path) -> dict[str, npt.NDArray[float]]:
    """
    Provided arguments from utils.parse_arguments and image dimensions, return the camera intrinsic matrix and distortion coefficients.
    :param calibration_file: path to calibration file (.yaml)
    :type calibration_file: Path
    :returns:
        - **K** (*npt.NDArray[float]*): camera intrinsic matrix
        - **D** (*npt.NDArray[float]*): distortion coefficients
    :rtype: dict[str, npt.NDArray[float]]
    """
    # If a calibration file has been given (a ROS camera_info yaml style file)
    calibration_path = Path(calibration_file).resolve(True)
    c = dict()
    with open(calibration_path, "r") as f:
        calib_data = yaml.safe_load(f)
    if 'camera_matrix' in calib_data:
        c.update({"K": np.array(calib_data['camera_matrix']['data']).reshape(
            (calib_data['camera_matrix']['rows'],
            calib_data['camera_matrix']['cols']))})
    else:
        logger.warning(f"No camera_matrix found in {str(calibration_path)}", UserWarning)
        return None
    if 'distortion_coefficients' in calib_data:
        c.update({"D": np.array(calib_data['distortion_coefficients']['data']).reshape(
            (calib_data['distortion_coefficients']['rows'],
            calib_data['distortion_coefficients']['cols']))})
    else:
        logger.warning(f"No distortion_coefficients found in {str(calibration_path)}", UserWarning)
        c.update({"D": np.zeros(5)})
    return c


def parse_args() -> argparse.Namespace:
    """
    Parse commandline arguments using argparse.
    :return: *args* - parsed arguments object.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Path to video file.")
    parser.add_argument("project", type=Path,
                        help="Path to directory where output mosaics are to be saved.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing database entries.")
    parser.add_argument("--force-cpu", action="store_true", help="Disable GPU operations.")
    parser.add_argument("--force-cuda-off", action="store_false", help="Disable CUDA operations.")
    parser.add_argument("--force-cudacodec-off", action="store_false", help="Disable CUDA codec operations.")

    # Video Player Arguments
    video_player_args = parser.add_argument_group("Video Player Params")
    start_time_group = video_player_args.add_mutually_exclusive_group()
    start_time_group.add_argument("--start-time-secs", metavar="START", type=float, help="Time (secs) to start from.")
    start_time_group.add_argument("--start-frame", metavar="START", type=int, help="Frame number to start from.")
    start_time_group.add_argument("--start-playtime", metavar="START", type=str, help="Playback time HH:MM:SS to start from.")
    finish_time_group = video_player_args.add_mutually_exclusive_group()
    finish_time_group.add_argument("--finish-time-secs", type=float, help="Time (secs) to finish at.")
    finish_time_group.add_argument("--finish-frame", type=int, help="Frame number to finish at.")
    finish_time_group.add_argument("--finish-playtime", type=str, help="Playback time HH:MM:SS to finish at.")
    video_player_args.add_argument("--frame-skip", type=int, default=None,
                        help="Number of frames to skip in video player.")

    # Preprocessing
    preprocessing_args = parser.add_argument_group("Preprocessing Params")
    preprocessing_args.add_argument("-c", "--calibration", type=Path, default=None,
                        help="Path to calibration file to apply undistortion.")

    # Stabilization Arguments
    stabilization_group = parser.add_argument_group(title="Stabilization Params")
    stabilization_group.add_argument("--orientation-path", type=Path, default=None,
                        help="Path to .csv file containing timestamped orientation measurements that transform world"
                             " to the camera frame.")
    stabilization_group.add_argument("--orientation-time-offset", type=float, default=None,
                                     help="Timestamp (secs) referenced to timestamp in orientation file where"
                                          "video starts (00:00:00).")

    # Feature Extraction
    feature_extraction_args = parser.add_argument_group("Feature Extraction Params")
    feature_extraction_args.add_argument("--feature-types", type=str, nargs="+",
                        choices=["ORB", "SIFT", "SURF", "BRISK", "KAZE", "AKAZE", "ALL"],
                        default=("ORB",), help="Set of features to use in registration.")
    feature_extraction_args.add_argument("--bovw-clusters", type=int, default=500, help="Number of bovw clusters to use in global feature descriptor.")
    feature_extraction_args.add_argument("--bovw-batchsize", type=int, default=1000, help="Batch size for bovw clustering.")
    feature_extraction_args.add_argument("--nn-top-k", type=int, default=5, help="Number of nearest neighbors to search for in global feature matching.")

    # Registration
    registration_args = parser.add_argument_group("Registration Params")
    registration_args.add_argument("--min-matches", type=int, default=10,
                        help="Minimum number of matches to proceed with registration.")
    registration_args.add_argument("--min-features", type=int, default=100,
                        help="Minimum number of features to detect in an image.")
    registration_args.add_argument("--homography-type", type=str, choices=["partial", "affine", "perspective"],
                        default="partial", help="Type of 2D homography estimation to perform.")
    registration_args.add_argument("--epsilon", type=float, default=1e-3, help="Homography determinant must be greater than this value to be considered valid.")

    # Mosaicking
    mosaicking_args = parser.add_argument_group("Mosaic Params")
    mosaicking_args.add_argument("--min-sequence-length", type=int, default=None, help="Minimum length of sequence to mosaic.")
    mosaicking_args.add_argument("--tile-size", type=int, default=1024,
                        help="Largest allowable size (width or height) for mosaic tile. Creates a new tile if it gets bigger.")
    mosaicking_args.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha blending scalar for merging new frames into mosaic. Default behaviour is to preserve existing canvas and append only new area.")
    mosaicking_args.add_argument("--keypoint-roi", action="store_true",
                        help="Only allow the convex hull of the inlier keypoints to be used in mosaic.")
    return parser.parse_args()


class DataReader(ABC):
    """
    Abstract base class for creating and managing a data reader.
    Provides a blueprint for data reader creation and resource release.

    **Methods**

    - :meth:`_create_reader`: Class method that initializes a specific data reader instance.
    - :meth:`release`: Releases any resources or handles associated with the data reader.
    """

    @classmethod
    @abstractmethod
    def _create_reader(cls: Type[T]) -> T:
        """
        Initializes and returns a new instance of the data reader.

        :returns: A new instance of a data reader implementing this class.
        :rtype: T
        """
        ...

    @abstractmethod
    def release(self):
        """
        Releases any resources or handles held by the data reader.
        """
        ...

    @abstractmethod
    def read(self) -> tuple[bool, str, npt.NDArray[np.uint8] | None]:
        ...

    @abstractmethod
    def peek_name(self) -> str:
        """
        Name of the next item to be read.
        """
        ...

    @abstractmethod
    def __getitem__(self, item):
        """
        For dictionary-like access
        """

    @abstractmethod
    def __iter__(self):
        """
        For iteration through data.
        """

    @abstractmethod
    def __len__(self):
        """
        For iteration through data.
        """

    @abstractmethod
    def __next__(self):
        """
        For iteration through data.
        """


class VideoPlayer(DataReader):
    """
    A video player class for reading and navigating through frames of a video file.

    **Parameters**

    - **video_path** (*Union[Path, os.PathLike]*): Path to the video file.
    - **frame_skip** (*int, optional*): Number of frames to skip when reading.
    - **start** (*Union[str, float, int], optional*): Starting point in the video, in seconds, frames, or HH:MM:SS.SS format.
    - **finish** (*Union[str, float, int], optional*): Ending point in the video, in seconds, frames, or HH:MM:SS.SS format.
    - **verbose** (*bool, optional*): Enables verbose logging.

    **Attributes**

    - **frame_skip** (*int*): Number of frames to skip when reading.
    - **frame_count** (*int*): Total frame count of the video.
    - **start_frame** (*int*): Starting frame index.
    - **finish_frame** (*int*): Ending frame index.
    - **width** (*int*): Width of the video frames.
    - **height** (*int*): Height of the video frames.
    - **fps** (*float*): Frames per second of the video.
    """

    def __init__(self, video_path: Union[Path, os.PathLike],
                 frame_skip: int = None,
                 start: Union[str, float, int] = None,
                 finish: Union[str, float, int] = None,
                 verbose: bool = False,
                 ):

        # Initialize the video reader
        self._video_path = Path(video_path).resolve(True)
        self._video_reader = self._create_reader()
        # Properties of the video
        self._fps = self.get(
            cv2.CAP_PROP_FPS)  # TODO: some videos don't have valid duration, so frame rate is undefined.
        self._width = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
        self._format_spec = "{" + f":0{len(str(self._frame_count))}d" + "}"
        self._frame_skip = frame_skip
        self._configure_pos(start, finish)  # !! Defines additional private attributes !!
        self._verbose = verbose
        if self._verbose:
            logger.info(f"Opened video {video_path}")

    @property
    def peek_name(self) -> str:
        """
        :return: A string representing the name of the video and frame position of the next frame to be read.
        """
        return f"{self._video_path.stem}-" + self._format_spec.format(int(self.get(cv2.CAP_PROP_POS_FRAMES)))

    @property
    def frame_skip(self) -> int:
        """
        :returns: Number of frames to skip when reading.
        :rtype: int
        """
        return self._frame_skip

    @property
    def frame_count(self) -> int:
        """
        :returns: Total number of frames in the video.
        :rtype: int
        """
        return self._frame_count

    @property
    def start_frame(self) -> int:
        """
        :returns: Starting frame index.
        :rtype: int
        """
        return self._start_frame

    @property
    def finish_frame(self) -> int:
        """
        :returns: Ending frame index.
        :rtype: int
        """
        return self._finish_frame

    @property
    def width(self) -> int:
        """
        :returns: Width of the video frames.
        :rtype: int
        """
        return self._width

    @property
    def height(self) -> int:
        """
        :returns: Height of the video frames.
        :rtype: int
        """
        return self._height

    @property
    def fps(self) -> float:
        """
        :returns: Frames per second of the video.
        :rtype: float
        """
        return self._fps

    @abstractmethod
    def _create_reader(self) -> Union[cv2.VideoCapture, 'cv2.cudacodec.VideoReader']:
        """
       Initializes and returns the video reader object.

       :returns: A VideoCapture or VideoReader object for reading video frames.
       :rtype: Union[cv2.VideoCapture, cv2.cudacodec.VideoReader]
       """
        ...

    @abstractmethod
    def get(self, propId: int) -> float:
        """
        Retrieves a specified property from the video reader.

        :param propId: Property ID to retrieve.
        :type propId: int
        :returns: The value of the specified property.
        :rtype: float
        """
        ...

    @abstractmethod
    def set(self, propId: int, value: float) -> bool:
        """
        Sets a specified property in the video reader.

        :param propId: Property ID to set.
        :type propId: int
        :param value: New value for the property.
        :type value: float
        :returns: True if the property was set successfully; False otherwise.
        :rtype: bool
        """
        ...

    def __getitem__(self, idx: int | str) -> tuple[bool, npt.NDArray[np.uint8] | None]:
        if isinstance(idx, str):
            match = re.search(r'\d+', idx)
            idx = int(match.group()) if match else None
            if idx is None:
                raise ValueError(f"Invalid index {idx}")
        self.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        return self.read()

    @abstractmethod
    def release(self):
        """
        Close the video reader object.
        """
        ...

    @staticmethod
    def _parse_time(playback_time: Union[str, float, int]) -> tuple[int, float]:
        """
        Parses a playback time specified as a string, frame number, or seconds.

        :param playback_time: Playback time in HH:MM:SS.SS format, frame number, or seconds.
        :type playback_time: Union[str, float, int]
        :returns: Property ID and time in milliseconds or frame index.
        :rtype: tuple[int, float]
        """
        # Playback time in HH:MM:SS.SS
        if isinstance(playback_time, str):
            time_pattern = re.compile(
                r'^(?P<hours>\d+):'  # 1 or more digits for hours
                r'(?P<minutes>[0-9]|[0-5][0-9]):'  # 1 or 2 digits for minutes
                r'(?P<seconds>[0-5][0-9](\.\d{1,3})?)$'
                # 2 digit int or 2-5 width float with 0-3 digits of precision for seconds
            )
            match = time_pattern.match(playback_time)
            assert match is not None, "Improper starting position expression HH+:MM:SS (optional .SSS)."
            start_msecs = (int(match.group("hours")) * 3600 + int(match.group("minutes")) * 60 + float(
                match.group("seconds"))) * 1000.0
            return cv2.CAP_PROP_POS_MSEC, start_msecs
        # Frame #
        elif isinstance(playback_time, int):
            return cv2.CAP_PROP_POS_FRAMES, playback_time
        # Run time
        elif isinstance(playback_time, float):
            start_msecs = 1000.0 * playback_time
            return cv2.CAP_PROP_POS_MSEC, start_msecs

    def read(self, frame: npt.NDArray=None) -> tuple[bool, int, str, npt.NDArray[np.uint8]| cv2.cuda.GpuMat | None]:
        """
        Reads the next frame from the video, considering any frame skip setting.

        :param frame: Frame buffer for output (optional).
        :returns: Tuple of success status, frame #, unique name, and the read frame.
        :rtype: tuple[bool, str, npt.NDArray[np.uint8] | cv2.cuda.GpuMat | None]
        """
        name = self.peek_name
        pos = int(self.get(cv2.CAP_PROP_POS_FRAMES))
        logger.debug(f"Read frame {pos}")
        if not self.frame_skip:
            ret, data = self._video_reader.read(frame)
            return ret, pos, name, data
        # If frame_skip specified, then include it
        next_frame = self.get(cv2.CAP_PROP_POS_FRAMES) + self.frame_skip
        ret, frame = self._video_reader.read(frame)
        self.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        return ret, pos, name, frame

    def _get_frame_pos(self, condition: tuple[int, float]) -> int:
        if condition[0] == cv2.CAP_PROP_POS_FRAMES:
            return int(condition[1])
        else:
            return int(condition[1] / 1000.0 * self.get(cv2.CAP_PROP_FPS))

    def _configure_pos(self, start: Union[str, int, float, None], finish: Union[str, int, float, None]):
        """
        Configures the starting and finishing positions of the video.

        :param start: Starting point in frames or seconds, or HH:MM:SS format.
        :type start: Union[str, int, float, None]
        :param finish: Finishing point in frames or seconds, or HH:MM:SS format.
        :type finish: Union[str, int, float, None]
        """
        # Set VideoCapture object to a starting position (either in seconds or the frame #)
        # VideoCapture reports current position (i.e. the frame # it is going to provide when "read" method is called)
        # as zero indexed.
        if start is not None:
            self._start_frame = self._get_frame_pos(self._parse_time(start))
        else:
            self._start_frame = 0
        if finish is not None:
            # TODO: add check to make sure finish condition is allowed.
            finish_frame = self._get_frame_pos(self._parse_time(finish))
            self._finish_frame = finish_frame if finish_frame < self._frame_count else self._frame_count - 1
        else:
            self._finish_frame = self._frame_count - 1
        assert self.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame), f"Could not set video start position to {start}"

    def __repr__(self):
        """
        Returns a formatted string showing the current frame position and total frame count.
        """
        fmt = "Next frame: {" + ":0{}d".format(len(str(self._frame_count))) + "} " + "of {}".format(
            self._finish_frame)
        return fmt.format(int(self.get(cv2.CAP_PROP_POS_FRAMES)))

    def _evaluate_stopping(self):
        """
        Checks if the current frame position has reached or exceeded the finish frame.

        :returns: True if the current frame is beyond the finish frame; False otherwise.
        :rtype: bool
        """
        return self.get(cv2.CAP_PROP_POS_FRAMES) > self._finish_frame

    def __iter__(self):
        """
        Returns an iterator for iterating over video frames.
        """
        return self

    def __len__(self):
        """
        Calculates the number of frames between the start and finish positions, considering frame skip.

        :returns: Number of frames in the defined playback range.
        :rtype: int
        """
        return len(range(*slice(self._start_frame, self._finish_frame, self._frame_skip).indices(self._frame_count)))

    def __next__(self) -> tuple[bool, int, str, npt.NDArray[np.uint8]| cv2.cuda.GpuMat | None]:
        """
        Retrieves the next frame, raising StopIteration if beyond the finish frame.

        :returns: Tuple of success status, frame #, name of frame and the read frame.
        :rtype: tuple[bool, np.ndarray]
        :raises StopIteration: When the playback reaches the finish frame.
        """
        # If the VideoPlayer still has frames to playback
        if not self._evaluate_stopping():
            return self.read()
        # End of Iteration
        raise StopIteration

    def __getstate__(self):
        # Create a copy of the instance’s state dictionary
        state = self.__dict__.copy()
        # Replace the attribute with the function to initialize it
        state['_video_reader'] = self._create_reader
        return state

    def __setstate__(self, state):
        # Restore the instance’s state
        self.__dict__.update(state)
        # Reinitialize self._video_reader
        self._video_reader = self._video_reader()


class CUDAVideoPlayer(VideoPlayer):

    def get(self, propId: int) -> float:
        return self._video_reader.get(propId)[1]

    def set(self, propId: int, value: float) -> bool:
        if propId == cv2.CAP_PROP_POS_FRAMES:
            return self._set_position(value)
        logger.warning("CUDAVideoPlayer.set doesn't work like CPUVideoPlayer")
        return self._video_reader.setVideoReaderProps(propId, value)

    def read(self, frame=None) -> tuple[bool, int, str, npt.NDArray[np.uint8]| cv2.cuda.GpuMat | None]:
        name = self.peek_name
        pos = int(self.get(cv2.CAP_PROP_POS_FRAMES))
        logger.debug(f"Read frame {pos}")
        if not self.frame_skip:
            ret, frame = self._video_reader.nextFrame(frame)
            return ret, pos, name, frame
        # If frame_skip specified, then include it
        next_frame = int(self.get(cv2.CAP_PROP_POS_FRAMES) + self.frame_skip)
        ret, frame = self._video_reader.nextFrame(frame)
        self._set_position(next_frame)
        return ret, pos, name, frame

    def release(self):
        self._video_reader = None

    def _create_reader(self, ) -> 'cv2.cudacodec.VideoReader':
        return cv2.cudacodec.createVideoReader(str(self._video_path))

    def _set_position(self, pos: int) -> bool:
        """Set the reader to read frame at position `pos`."""
        if pos < self.get(cv2.CAP_PROP_POS_FRAMES):
            # reset the video reader
            self._video_reader = self._create_reader()
        while self.get(cv2.CAP_PROP_POS_FRAMES) < pos and not self._evaluate_stopping():
            self._video_reader.grab()
        return True


class CPUVideoPlayer(VideoPlayer):

    def _create_reader(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(str(self._video_path))

    def get(self, propId: int) -> float:
        return self._video_reader.get(propId)

    def set(self, propId: int, value: float) -> bool:
        return self._video_reader.set(propId, value)

    def release(self):
        self._video_reader.release()
        self._video_reader = None


def load_orientations_pd(path: os.PathLike, args: argparse.Namespace) -> pd.DataFrame:
    """
    Given a path containing orientations, retrieve the orientations corresponding to a time offset between video and orientation data.
    """
    time_offset = args.time_offset if args.time_offset else args.sync_points[1] - args.sync_points[0]
    df = pd.read_csv(str(path), index_col="timestamp")
    df.index = df.index - time_offset
    return df[~df.duplicated()]


def load_orientation_slerp(orientation_file: Path, video_time_offset: float = 0.0) -> Slerp:
    """
    Spherical linear interpolation of orientation quaternions provided by csv-like file.
    """
    df = pd.read_csv(orientation_file)
    # Assuming your DataFrame is named df and the timestamp column is named 'timestamp'
    df = df.sort_values(by='ts', ascending=True)
    df["pt"] = df["ts"] - video_time_offset
    return Slerp(df['ts'], Rotation.from_quat(df.loc[:, ['qx', 'qy', 'qz', 'qw']]))

def convert_feature_keypoints(features: dict[str, dict[str, Union[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]]]) -> dict[str, dict[str, Union[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]]]:
    """Convert numpy keypoints into cv2.KeyPoint objects, or reverse."""
    output_dict = dict()
    for feature_type, feat in features.items():
        kp = feat["keypoints"]
        feat.update({"keypoints": cv2.KeyPoint.convert(kp)})
        output_dict.update({feature_type: feat})
    return output_dict

def get_descriptor_tuples(features: dict[str, dict[str, Union[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]]], feature_types: Sequence[str]) -> Sequence[npt.NDArray[np.float32]]:
    """"""
    return tuple(features[feature_type]["descriptors"] for feature_type in feature_types)

def get_keypoint_tuples(features: dict[str, dict[str, Union[Sequence[cv2.KeyPoint], npt.NDArray[np.float32]]]], feature_types: Sequence[str]) -> Sequence[npt.NDArray[np.float32]]:
    """"""
    return tuple(features[feature_type]["keypoints"] for feature_type in feature_types)
