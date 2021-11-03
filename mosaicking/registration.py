import sys

import cv2
import numpy as np
import imutils
from skimage import util
import copy


def get_matches(descriptors1: np.ndarray, descriptors2: np.ndarray, matcher: cv2.DescriptorMatcher, minmatches: int):
    # New image prepared and enough features were found, find matches.
    # Match descriptors between previous and new
    knn_matches = matcher.knnMatch(descriptors1.astype(np.float32), descriptors2.astype(np.float32), 2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in knn_matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return minmatches < len(good), good


def get_features(img: np.ndarray, fdet: cv2.Feature2D, mask=None):
    """
    Given a feature detector, obtain the features found in the image.
    """
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return fdet.detectAndCompute(img, mask)


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
            sys.stderr.write("WARNING: Gradient Explosion. Is the Image Rapidly Zooming In/Out? Gradient clipping is suppressed to allow continuity, consider increasing gradient clip argument.")
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

    t = [-xmin, -ymin]  # translation of the upper left corner of the image
    A[:, -1] = A[:, -1] + t  # translation homography
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
            sys.stderr.write("WARNING: Gradient Explosion. Is the Image Rapidly Zooming In/Out? Gradient clipping is suppressed to allow continuity, consider increasing gradient clip argument.")
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

    t = [-xmin, -ymin]  # translation of the upper left corner of the image
    A[:, -1] = A[:, -1] + t  # translation homography
    return A, (xmin, xmax), (ymin, ymax)


def alignImages(im1: np.ndarray, im2: np.ndarray, detector: cv2.Feature2D, last_tf=None):
    """im1: source image (to be registered onto im2)
    img2: destination image
    detector: some kind of feature detector"""
    im1 = util.img_as_ubyte(im1)
    im2 = util.img_as_ubyte(im2)

    ## Turn frames to grayscale
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    ##  Get the binary masks from the grayscaled frames
    ret1, mask1 = cv2.threshold(gray1, 1, 255, cv2.THRESH_BINARY)
    ret2, mask2 = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)

    ## Find features between frames using binary masks too
    kp1, descriptors1 = detector.detectAndCompute(gray1, mask1)
    kp2, descriptors2 = detector.detectAndCompute(gray2, mask2)

    if len(kp1) == 0 or len(kp2) == 0:
        return False, None

    # Initializes parameters for Flann-based matcher
    flann_index_kdtree = 0
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)

    # Initializes the Flann-based matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Computes matches using Flann matcher
    matches = flann.knnMatch(descriptors2, descriptors1, k=2)
    good = []
    for j, (m, n) in enumerate(matches):
        if m.distance < 0.77 * n.distance:
            good.append(m)

    print (str(len(good)) + " Matches were Found")

    ## Copy the good matches in matches
    matches = copy.copy(good)

    ## Define source and destination points between the images based on the good matches
    src_pts = np.array([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.array([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    ##  Find homography matrix
    H, status = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    ## Find height and width of both frames
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    ## Find the corners of frame 1 and frame 2
    corners1 = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)

    ## Warp the corners of im2
    warpedCorners2 = cv2.perspectiveTransform(corners2, H)  # The transformed corners of img 2

    # Consider the corner points of both mosaic and input image
    corners = np.concatenate((corners1, warpedCorners2), axis=0)

    # Find the min and max offset for the bounding box
    bx, by, bwidth, bheight = cv2.boundingRect(corners)
    # Calculate translation
    translation = np.float32([[1, 0, -bx],
                              [0, 1, -by],
                              [0, 0, 1]])

    ## Translate img1 to locate within combined mosaic
    warpedResImg = cv2.warpPerspective(im1, translation, (bwidth, bheight),
                                       flags=cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS)
    if len(good) >= 100:
        # Warping will be with the dot product of the translation and the homography matrix
        fullTransformation = np.dot(translation, H)
        # last_tf places img2 within the mosaic
        last_tf = fullTransformation
        # apply translation and rotation to im2
        warpedImage2 = cv2.warpPerspective(im2, fullTransformation, (bwidth, bheight),
                                           flags=cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS)
    else:
        return False, im1, last_tf

    result = np.where(warpedImage2 != 0, warpedImage2, warpedResImg)

    # transform the panorama image to grayscale and threshold it
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Find contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)

    # crop the image to the bbox coordinates
    result = result[y:y + h, x:x + w]

    return True, result, last_tf