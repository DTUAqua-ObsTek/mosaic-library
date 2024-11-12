import re

import cv2

import logging

# Create a logger for the package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


try:
    from ._version import __version__
except ImportError:
    __version__ = "dev"  # Fallback version during development


def check_cuda() -> bool:
    """
    Checks if OpenCV is built with NVIDIA CUDA Toolkit Support.
    :return:
        **has_cuda** (bool): flag for CUDA support
    :rtype: bool
    """
    build_info = cv2.getBuildInformation()
    match = re.search(r'NVIDIA CUDA:\s+(YES|NO)', build_info)
    if match:
        cuda_status = match.group(1)
        if cuda_status == "YES":
            logger.info("CUDA is enabled in your OpenCV build.")
            return True
        else:
            logger.info("CUDA is not enabled in your OpenCV build.")
            return False
    logger.info("CUDA status not found in the build information.")
    return False

HAS_CUDA = check_cuda()

def check_cudacodec() -> bool:
    """
    Checks if OpenCV is built with NVIDIA CUVID Encode/Decode Support.
    :return:
        **has_cuvid** (bool): flag for cuvid support
    :rtype: bool
    """
    if HAS_CUDA:
        if hasattr(cv2, "cudacodec"):
            logger.info("CUDA codec is enabled in your OpenCV build.")
            return True
        logger.info("CUDA codec is not enabled in your OpenCV build.")
        return False
    logger.info("CUDA codec is not enabled in your OpenCV build. Check if built with CUDA.")
    return False

HAS_CODEC = check_cudacodec()

import importlib.resources

def load_schema_sql():
    # Access schema.sql content as a string
    with importlib.resources.open_text('mosaicking.resources', 'schema.sql') as f:
        schema_sql = f.read()
    return schema_sql
