import numpy as np
from typing import Sequence
import cv2

import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


def calculate_tiles(bounding_box: tuple[int, int, int, int], tile_size: tuple[int, int]) -> Sequence[tuple[int, int, int, int]]:
    """
    Calculate the number of tiles and their bounding areas based on the composite image bounding box and tile size.

    Args:
        bounding_box (tuple): The bounding box of the composite image (min_x, min_y, width, height).
        tile_size (tuple): The size of each tile (width, height).

    Returns:
        tuple: A tuple of bounding areas for each tile [(min_x, min_y, max_x, max_y), ...].
    """
    min_x, min_y, width, height = bounding_box
    max_x = min_x + width
    max_y = min_y + height
    tile_width, tile_height = tile_size

    x_tiles = int(np.ceil((width) / tile_width))
    y_tiles = int(np.ceil((height) / tile_height))

    tiles = []
    for i in range(x_tiles):
        for j in range(y_tiles):
            tile_min_x = min_x + i * tile_width
            tile_min_y = min_y + j * tile_height
            tile_max_x = min(tile_min_x + tile_width, max_x)
            tile_max_y = min(tile_min_y + tile_height, max_y)
            tiles.append((tile_min_x, tile_min_y, tile_max_x, tile_max_y))

    return tiles


def assign_image_to_tiles(warped_corners, tiles):
    """
    Assign a warped image to specific tiles based on whether its corners are within a tile's bounding area.

    Args:
        warped_corners (np.array): The warped corners of the image.
        tiles (list): The list of tile bounding areas [(min_x, min_y, max_x, max_y), ...].

    Returns:
        dict: A dictionary mapping each tile to the corresponding homographies for that tile.
    """
    assignments = []
    for idx, (min_x, min_y, max_x, max_y) in enumerate(tiles):
        if np.any((warped_corners[:, 0] >= min_x) & (warped_corners[:, 0] < max_x) &
                  (warped_corners[:, 1] >= min_y) & (warped_corners[:, 1] < max_y)):
            assignments.append(idx)
    return tuple(assignments)


def calculate_translation_homography(homography, tile_origin):
    """
    Calculate the translation transformation required to shift the image into tile local coordinates.

    Args:
        homography (np.array): The homography matrix.
        tile_origin (tuple): The origin of the tile (min_x, min_y).

    Returns:
        np.array: The translated homography matrix.
    """
    translation = np.array([
        [1, 0, -tile_origin[0]],
        [0, 1, -tile_origin[1]],
        [0, 0, 1]
    ])
    return translation @ homography


def process_images(homographies, images, bounding_box, tile_size):
    """
    Process the images by splitting the composite image into tiles and performing the required transformations.

    Args:
        homographies (list): List of homography matrices for each image.
        images (list): List of images to be warped.
        bounding_box (tuple): The bounding box of the composite image (min_x, min_y, max_x, max_y).
        tile_size (tuple): The size of each tile (width, height).

    Returns:
        dict: A dictionary mapping each tile to the list of warped images.
    """
    tiles = calculate_tiles(bounding_box, tile_size)
    tile_images = {i: [] for i in range(len(tiles))}

    for homography, image in zip(homographies, images):
        h, w = image.shape[:2]
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ])
        warped_corners = (homography @ corners.T).T
        warped_corners /= warped_corners[:, 2][:, None]

        assignments = assign_image_to_tiles(warped_corners[:, :2], tiles)

        for tile_idx, _ in assignments.items():
            tile_origin = tiles[tile_idx][:2]
            translated_homography = calculate_translation_homography(homography, tile_origin)
            warped_image = cv2.warpPerspective(image, translated_homography, tile_size)
            tile_images[tile_idx].append(warped_image)

    return tile_images
