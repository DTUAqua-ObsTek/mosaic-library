import re

import cv2

try:
    from ._version import __version__
except ImportError:
    __version__ = "dev"  # Fallback version during development


def check_cuda():
    build_info = cv2.getBuildInformation()
    match = re.search(r'NVIDIA CUDA:\s+(YES|NO)', build_info)
    if match:
        cuda_status = match.group(1)
        if cuda_status == "YES":
            print("CUDA is enabled in your OpenCV build.")
            return True
        else:
            print("CUDA is not enabled in your OpenCV build.")
            return False
    print("CUDA status not found in the build information.")
    return False

HAS_CUDA = check_cuda()

def check_cudacodec():
    if HAS_CUDA:
        build_info = cv2.getBuildInformation()
        if "NVCUVID" in build_info:
            print("CUDA codec is enabled in your OpenCV build.")
            return True
        else:
            print("CUDA codec is not enabled in your OpenCV build.")
            return False
    return False

HAS_CODEC = check_cudacodec()
