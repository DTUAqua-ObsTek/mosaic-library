import sys

import cv2
import numpy as np
from typing import Union


def get_matches(descriptors1: Union[np.ndarray, list], descriptors2: Union[np.ndarray, list], matcher: cv2.DescriptorMatcher, minmatches: int):
    if type(descriptors1) is not list:
        descriptors1 = [descriptors1]
    if type(descriptors2) is not list:
        descriptors2 = [descriptors2]
    mlength = 0
    nlength = 0
    good = []
    for d1, d2 in zip(descriptors1, descriptors2):
        matches = matcher.knnMatch(d1.astype(np.float32), d2.astype(np.float32), 2)
        for m, n in matches:
            m.queryIdx = m.queryIdx + mlength
            m.trainIdx = m.trainIdx + nlength
            n.queryIdx = n.queryIdx + nlength
            n.trainIdx = n.trainIdx + mlength
            if m.distance < 0.7 * n.distance:
                good.append(m)
        mlength += d1.shape[0]
        nlength += d2.shape[0]
    return minmatches < len(good), good


def get_features(img: np.ndarray, fdet: cv2.Feature2D, mask=None):
    """
    Given a feature detector, obtain the features found in the image.
    """
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return fdet.detectAndCompute(img, mask)


def get_alignment(src_pts: np.ndarray, src_shape: tuple, dst_pts: np.ndarray, dst_shape: tuple, homography: str = "similar", gradient: float = 0.0):
    assert homography in ["rigid", "similar", "affine", "perspective"], "Homography can be of type similar, affine, or perspective"
    # Update the homography from current image to mosaic
    if homography == "rigid":
        A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1)
        # Remove rotation components
        theta = np.arctan2(A[1,0], A[0, 0])
        s = A[0, 0] / np.cos(theta) if np.cos(theta) != 0 else A[1, 0] / np.sin(theta)
        A[:, :2] = s * np.eye(2)
    elif homography == "similar":
        A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)#, method=cv2.RANSAC, ransacReprojThreshold=1)
        # # Remove shear effect from affine transform (according to: https://math.stackexchange.com/a/3521141)
        # sx = np.sqrt(A[0, 0]**2+A[1,0]**2)
        # theta = np.arctan2(A[1,0],A[0,0])
        # msy = A[0,1]*np.cos(theta) + A[1,1]*np.sin(theta)
        # if np.abs(np.sin(theta)) > 0:
        #     sy = (msy*np.cos(theta)-A[0,1]) / np.sin(theta)
        # else:
        #     sy = (A[1,1] - msy*np.sin(theta)) / np.cos(theta)
        # m = msy / sy
        # rot = np.array([[np.cos(theta), -np.sin(theta)],
        #                      [np.sin(theta), np.cos(theta)]])
        # shear = np.eye(2)
        # scale = np.array([[sx, 0],
        #                   [0, sy]])
        # A[:2,:2] = rot@shear@scale
    elif homography == "affine":
        A, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1)
    else:
        A, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1)
    A = np.concatenate((A, np.array([[0, 0, 1]])), axis=0) if A.size < 9 else A  # convert to a homogeneous form
    # Warp the image coordinates
    ugrid = np.arange(0, src_shape[1] - 1)
    vgrid = np.arange(0, src_shape[0] - 1)
    uu, vv = np.meshgrid(ugrid, vgrid, indexing='ij')
    grid = np.stack((uu.flatten(), vv.flatten(), np.ones_like(vv.flatten())), 0)
    warp_dst = A @ grid
    warp_dst = warp_dst[:2, :] / warp_dst[-1, :]
    if gradient > 0:
        grad = np.gradient(warp_dst, axis=1)
        idx = np.sqrt((grad ** 2).sum(axis=0)) < gradient
        if not idx.any():
            sys.stderr.write("WARNING: Gradient Explosion. Is the Image Rapidly Zooming In/Out? Gradient clipping is suppressed to allow continuity, consider increasing gradient clip argument.\n")
        else:
            warp_dst = warp_dst[:, idx]

    # Get the corners of the mosaic image in homogeneous coords (x,y,w=1)
    dst_crn = np.array([[0, dst_shape[1], dst_shape[1], 0],
                        [0, 0, dst_shape[0], dst_shape[0]]], float)

    # Concatenate the mosaic and warped corner coordinates
    pts = np.concatenate([dst_crn, warp_dst], axis=1)

    # Round to pixel centers
    xmin, ymin = np.int32(pts.min(axis=1) + 0.5)
    xmax, ymax = np.int32(pts.max(axis=1) + 0.5)

    t = [-xmin, -ymin]  # C of the upper left corner of the image
    A[:2, -1] = A[:2, -1] + t  # C homography
    return A, (xmin, xmax), (ymin, ymax)


def get_similarity_alignment(src_pts: np.ndarray, src_shape: tuple, dst_pts: np.ndarray, dst_shape: tuple, gradient: float = 0.0):
    # Update the homography from current image to mosaic
    #A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)

    # # Get the corners of the current image in homogeneous coords (x,y,w=1)
    # src_crn = np.array([[0, img.shape[1], img.shape[1], 0],
    #                     [0, 0, img.shape[0], img.shape[0]],
    #                     [1, 1, 1, 1]], float)

    # Warp a grid
    xgrid = np.arange(0, src_shape[1] - 1)
    ygrid = np.arange(0, src_shape[0] - 1)
    xx, yy = np.meshgrid(xgrid, ygrid, indexing='ij')
    grid = np.stack((xx.flatten(), yy.flatten(), np.ones_like(yy.flatten())), 0)
    warp_dst = A @ grid
    # warp_dst = warp_dst[:2, :] / warp_dst[-1, :]
    if gradient > 0:
        grad = np.gradient(warp_dst, axis=1)
        idx = np.sqrt((grad ** 2).sum(axis=0)) < gradient
        if not idx.any():
            sys.stderr.write("WARNING: Gradient Explosion. Is the Image Rapidly Zooming In/Out? Gradient clipping is suppressed to allow continuity, consider increasing gradient clip argument.\n")
        else:
            warp_dst = warp_dst[:, idx]

    # Get the corners of the mosaic image in homogeneous coords (x,y,w=1)
    dst_crn = np.array([[0, dst_shape[1], dst_shape[1], 0],
                        [0, 0, dst_shape[0], dst_shape[0]],
                        [1, 1, 1, 1]], float)

    # Concatenate the mosaic and warped corner coordinates
    pts = np.concatenate([dst_crn[:2, :], warp_dst], axis=1)

    # Round to pixel centers
    # xmin, ymin, width, height = cv2.boundingRect(pts.T.astype(np.float32))
    # xmax = xmin + width - 1
    # ymax = ymin + height - 1
    xmin, ymin = np.int32(pts.min(axis=1) - 0.5)
    xmax, ymax = np.int32(pts.max(axis=1) + 0.5)

    t = [-xmin, -ymin]  # C of the upper left corner of the image
    A[:, -1] = A[:, -1] + t  # C homography
    return A, (xmin, xmax), (ymin, ymax)


def get_affine_alignment(src_pts: np.ndarray, src_shape: tuple, dst_pts: np.ndarray, dst_shape: tuple, gradient: float = 0.0):
    # Update the homography from current image to mosaic
    #A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    #A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    A, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.LMEDS)

    # # Get the corners of the current image in homogeneous coords (x,y,w=1)
    # src_crn = np.array([[0, img.shape[1], img.shape[1], 0],
    #                     [0, 0, img.shape[0], img.shape[0]],
    #                     [1, 1, 1, 1]], float)

    # Warp a grid
    xgrid = np.arange(0, src_shape[1] - 1)
    ygrid = np.arange(0, src_shape[0] - 1)
    xx, yy = np.meshgrid(xgrid, ygrid, indexing='ij')
    grid = np.stack((xx.flatten(), yy.flatten(), np.ones_like(yy.flatten())), 0)
    warp_dst = A @ grid
    # warp_dst = warp_dst[:2, :] / warp_dst[-1, :]
    if gradient > 0:
        grad = np.gradient(warp_dst, axis=1)
        idx = np.sqrt((grad ** 2).sum(axis=0)) < gradient
        if not idx.any():
            sys.stderr.write("WARNING: Gradient Explosion. Is the Image Rapidly Zooming In/Out? Gradient clipping is suppressed to allow continuity, consider increasing gradient clip argument.\n")
        else:
            warp_dst = warp_dst[:, idx]

    # Get the corners of the mosaic image in homogeneous coords (x,y,w=1)
    dst_crn = np.array([[0, dst_shape[1], dst_shape[1], 0],
                        [0, 0, dst_shape[0], dst_shape[0]],
                        [1, 1, 1, 1]], float)

    # Concatenate the mosaic and warped corner coordinates
    pts = np.concatenate([dst_crn[:2, :], warp_dst], axis=1)

    # Round to pixel centers
    # xmin, ymin, width, height = cv2.boundingRect(pts.T.astype(np.float32))
    # xmax = xmin + width - 1
    # ymax = ymin + height - 1
    xmin, ymin = np.int32(pts.min(axis=1) - 0.5)
    xmax, ymax = np.int32(pts.max(axis=1) + 0.5)

    t = [-xmin, -ymin]  # C of the upper left corner of the image
    A[:, -1] = A[:, -1] + t  # C homography
    return A, (xmin, xmax), (ymin, ymax)
