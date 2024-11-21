from typing import Sequence

import numpy as np


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
