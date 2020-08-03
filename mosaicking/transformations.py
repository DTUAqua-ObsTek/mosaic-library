import numpy as np
import cv2


## euler rotation methods
# Matrix for Yaw-rotation about the Z-axis
def R_z(phi):
    R = np.array([[np.cos(phi),   -np.sin(phi), 0],
         [np.sin(phi),   np.cos(phi),   0],
         [0,  0, 1]])
    return R


# Matrix for Pitch-rotation about the Y-axis
def R_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
         [0, 1, 0,],
         [-np.sin(theta), 0, np.cos(theta)]])
    return R


# Matrix for Roll-rotation about the X-axis
def R_x(phi):
    R = np.array([[1,  0,           0],
         [0,  np.cos(phi),   -np.sin(phi)],
         [0,  np.sin(phi),   np.cos(phi)]])
    return R


def euler_affine_rotate(img: np.ndarray, euler: np.array, degrees: bool=False):
    """img: numpy array image
    euler: numpy array euler angles in Yaw, Pitch, Roll order"""
    if degrees:
        euler = np.deg2rad(euler)
    R = np.matmul(R_y(euler[1]), np.matmul(R_x(euler[2]), R_z(euler[0])))
    height, width, channels = img.shape
    src = np.array([[0, 0, 0],
                    [0, height-1, 0],
                    [width-1, height-1, 0],
                    [width-1, 0, 0]], dtype="float32")
    # the destination points are rotated by R
    dst = np.matmul(src,R)[:,0:2].astype("float32")
    src = src[:,0:2].astype("float32")
    bx, by, bwidth, bheight = cv2.boundingRect(dst)
    h = cv2.getAffineTransform(src[0:3,0:2], dst[0:3,:])
    h[:,-1] = [-bx, -by]
    img_dst = cv2.warpAffine(img, h, (bwidth, bheight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return img_dst


## Bird View method- get a top down view of the frames
def bird_view(image, pitch=45):
    ## Crop image
    image = image[0:(image.shape[0] - 100), 0:(image.shape[1] - 100)]

    IMAGE_H = image.shape[0]
    IMAGE_W = image.shape[1]
    #    first_frame_cro = first_frame[100:(first_frame.shape[0]-100),100:(first_frame.shape[1]-100)]

    # Assume that focus length is equal half image width

    # Translation of images on the black canvases
    translation = np.float32(([1, 0, -1 * IMAGE_W / 2], [0, 1, -1 * IMAGE_H], [0, 0, 1]))
    FOCUS = IMAGE_W / 2  ## Focus needs to stay the same height of the seabed-camera
    warped_img = None
    pRad = pitch * np.pi / 180
    sinPt = np.sin(pRad)
    cosPt = np.cos(pRad)
    Yp = IMAGE_H * sinPt
    Zp = IMAGE_H * cosPt + FOCUS
    Xp = -IMAGE_W / 2
    XDiff = Xp * FOCUS / Zp + IMAGE_W / 2
    YDiff = IMAGE_H - Yp * FOCUS / Zp
    # Vary upper source points
    src = np.float32([[0, IMAGE_H - 1], [IMAGE_W - 1, IMAGE_H - 1], [XDiff, YDiff], [IMAGE_W - 1, YDiff]]).reshape(-1,
                                                                                                                   1, 2)
    dst = np.float32([[0, IMAGE_H - 1], [IMAGE_W - 1, IMAGE_H - 1], [0, 0], [IMAGE_W - 1, 0]]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    warpedCorners = cv2.perspectiveTransform(src, H)  # The transformation matrix
    [xMin, yMin] = np.int32(warpedCorners.min(axis=0).ravel() - 0.5)  # new dimensions
    [xMax, yMax] = np.int32(warpedCorners.max(axis=0).ravel() + 0.5)
    translation = np.array(
        ([1, 0, -1 * xMin], [0, 1, -1 * yMin], [0, 0, 1]))  # must translate image so that all of it is visible
    fullTransformation = np.dot(translation, H)  # compose warp and translation in correct order
    #    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
    warped_img = cv2.warpPerspective(image, fullTransformation, (2 * IMAGE_W, 2 * IMAGE_H),
                                     flags=cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS)  # Image warping

    ## Numpy slicing of the original Image and the warped Image
    #    warped_img = warped_img[IMAGE_H /2:(warped_img.shape[0]-IMAGE_H /2),IMAGE_W/2:(warped_img.shape[1]-IMAGE_W/2)]
    return warped_img