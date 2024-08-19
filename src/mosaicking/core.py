from typing import Sequence, Union, Any
import numpy as np
import numpy.typing as npt
import networkx as nx
from dataclasses import dataclass
import cv2

# TODO: the graph representation could (not sure if they should be placed this far down)
#  1. compute inverse homography when edge added.
#  2. use feature matching to find new edges when queried
#  3. provide the absolutely homography chain when provided a path query.
#  4. calculate the confidence of a homography by reprojecting keypoints from source to destination,
#  cumulative summation of error and normalization.

@dataclass
class ImageNode:
    features: dict[str, dict[str, Union[Sequence[cv2.KeyPoint], Union[None, npt.NDArray[np.float32]]]]]  # features = {feature_type: {keypoints: [cv2.KeyPoint], descriptors: np.ndarray}}

    def __post_init__(self):
        # Convert cv2.KeyPoint sequences to NumPy arrays
        for feature_type, feature_data in self.features.items():
            if "keypoints" in feature_data and isinstance(feature_data["keypoints"], Sequence):
                keypoints = feature_data["keypoints"]
                # Convert keypoints to np.ndarray using cv2.KeyPoint_convert
                feature_data["keypoints"] = cv2.KeyPoint.convert(keypoints)


@dataclass
class Edge:
    matches: Sequence[cv2.DMatch]
    homography: np.ndarray

@dataclass
class DMatch:
    queryIdx: int  # Index of the descriptor in the query set
    trainIdx: int  # Index of the descriptor in the train set
    imgIdx: int    # Index of the train image (in case of multiple images)
    distance: float  # Distance between descriptors


class ImageGraph(nx.DiGraph):
    """A graph representation for registration of images in a dataset."""

    def add_image(self, index: int, data: ImageNode):
        """
        Add an image node to the graph.
        """
        self.add_node(index, data=data)

    def add_registration(self, image1: int, image2: int, registration: Edge):
        """
        Add an edge representing the link between two images.

        Args:
        - image1: The index of the first image.
        - image2: The index representation of the second image.
        - edge: A Edge representing the registration between the two images.
        """
        super().add_edge(image1, image2, registration=registration)

    def get_matches(self, image_id1: int, image_id2: int) -> Sequence[cv2.DMatch]:
        """
        Get the matches between two images.
        """
        return self[image_id1][image_id2]['registration'].matches

    def get_homography(self, image_id1: int, image_id2: int) -> np.ndarray:
        """
        Get the homography matrix between two images.
        """
        return self[image_id1][image_id2]['registration'].homography


def homogeneous_translation(x: float, y: float) -> np.ndarray:
    out = np.eye(3)
    out[[0, 1], 2] = [x, y]
    return out


def homogeneous_scaling(*sf: Sequence[float]) -> np.ndarray:
    assert len(sf) < 2, "Maximum two scaling factors allowed."
    out = np.eye(3)
    out[[0, 1], [0, 1]] = sf
    return out


def concatenate_with_slices(ndarrays: Sequence[np.ndarray], axis: int = 0) -> tuple[np.ndarray, Sequence[slice]]:
    """
    Concatenate a list of ndarrays and return the concatenated array
    and a list of slices indicating the start and end indices of each ndarray.

    Parameters:
    - ndarray_list: List of ndarrays to concatenate.

    Returns:
    - concatenated_array: The concatenated ndarray.
    - slices: List of tuples with start and end indices for each ndarray.
    """
    concatenated_array = np.concatenate(ndarrays, axis=axis)
    slices = []
    start_index = 0

    for array in ndarrays:
        end_index = start_index + array.shape[axis]
        slices.append(slice(start_index, end_index))
        start_index = end_index

    return concatenated_array, slices


def inverse_K(K: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of a zero-skew pinhole-camera projection matrix.
    """
    K_inv = np.eye(*K.shape)
    K_inv[0, 0] = K[1, 1]
    K_inv[1, 1] = K[0, 0]
    K_inv[0, 1] = -K[0, 1]
    K_inv[0, 2] = K[1, 2] * K[0, 1] - K[0, 2] * K[1, 1]
    K_inv[1, 2] = -K[1, 2] * K[0, 0]
    K_inv[2, 2] = K[0, 0] * K[1, 1]
    return 1 / (K[0, 0] * K[1, 1]) * K_inv


def group_sequences_length_threshold(sequences: Sequence[Sequence[Any]], length: int) -> tuple[tuple[Sequence[Any], ...], ...]:
    result = []
    current_group = []

    for s in sequences:
        if len(s) >= length:
            current_group.append(s)
        else:
            if current_group:
                result.append(tuple(current_group))
                current_group = []

    # Add the last group if it exists
    if current_group:
        result.append(tuple(current_group))

    return tuple(result)
