import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


# TODO: the graph representation could (not sure if they should be placed this far down)
#  1. compute inverse homography when edge added.
#  2. use feature matching to find new edges when queried
#  3. provide the absolutely homography chain when provided a path query.
#  4. calculate the confidence of a homography by reprojecting keypoints from source to destination,
#  cumulative summation of error and normalization.


