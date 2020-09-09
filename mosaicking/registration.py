import cv2
import numpy as np
import imutils
from skimage import util
import copy


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
    warpedCorners2 = cv2.perspectiveTransform(corners2, H)  # The transformation matrix

    # Consider the corner points of both mosaic and input image
    corners = np.concatenate((corners1, warpedCorners2), axis=0)

    # # Find the min and max offset for the bounding box
    # [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    # [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    # # Calculate translation
    # translation = np.float32(([1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]))
    bx, by, bwidth, bheight = cv2.boundingRect(corners)
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