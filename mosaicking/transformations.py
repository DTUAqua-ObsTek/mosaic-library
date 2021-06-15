import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


def calculate_friendly_affine(A: np.ndarray, width: int, height: int, gradient_clip: float):
    # Warp the corners of the image
    xgrid = np.arange(0, width - 1)
    ygrid = np.arange(0, height - 1)
    xx, yy = np.meshgrid(xgrid, ygrid, indexing='ij')
    grid = np.stack((xx.flatten(), yy.flatten(), np.ones_like(yy.flatten())), 0)
    warp_grid = A @ grid
    pts = warp_grid[:2, :] / warp_grid[-1, :]
    if gradient_clip > 0:
        grad = np.gradient(pts, axis=1)
        idx = np.sqrt((grad ** 2).sum(axis=0)) < gradient_clip
        pts = pts[:, idx]
    # Round to pixel centers
    xmin, ymin = np.int32(pts.min(axis=1) - 0.5)
    xmax, ymax = np.int32(pts.max(axis=1) + 0.5)
    T1 = np.array([[1, 0, -xmin],
                   [0, 1, -ymin]])
    A = np.linalg.multi_dot([T1, A])


def calculate_homography(K: np.ndarray, width: int, height: int, euler_x: float, euler_y: float, euler_z: float, gradient_clip: float=0.0):
    if K.shape[1] < 4:
        K = np.concatenate((K, np.zeros((3, 1))), axis=1)
    Kinv = np.zeros((4, 3))
    Kinv[:3, :3] = np.linalg.inv(K[:3, :3]) * (K[0, 0] * K[1, 1])
    Kinv[-1, :] = [0, 0, 1]
    R = euler_rotation(euler_x, euler_y, euler_z)
    # Translation matrix
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    # Overall homography matrix
    H = np.linalg.multi_dot([K, R, T, Kinv])
    # Warp the corners
    xgrid = np.arange(0, width - 1)
    ygrid = np.arange(0, height - 1)
    xx, yy = np.meshgrid(xgrid, ygrid, indexing='ij')
    grid = np.stack((xx.flatten(), yy.flatten(), np.ones_like(yy.flatten())), 0)
    warp_grid = H @ grid
    pts = warp_grid[:2, :] / warp_grid[-1, :]
    if gradient_clip > 0:
        grad = np.gradient(pts, axis=1)
        idx = np.sqrt((grad ** 2).sum(axis=0)) < gradient_clip
        pts = pts[:, idx]
    # Round to pixel centers
    xmin, ymin = np.int32(pts.min(axis=1) - 0.5)
    xmax, ymax = np.int32(pts.max(axis=1) + 0.5)
    T1 = np.array([[1, 0, -xmin],
                   [0, 1, -ymin],
                   [0, 0, 1]])
    H = np.linalg.multi_dot([T1, K, R, T, Kinv])
    return H, (xmax - xmin, ymax - ymin), (xmin, xmax), (ymin, ymax)


def apply_rotation(img: np.ndarray, K: np.ndarray, rotation: np.ndarray, keypoints: list, mask: np.ndarray = None, gradient_clip: float = 0.0):
    eulers = Rotation.from_dcm(rotation).as_euler("xyz")
    H, bounds = calculate_homography(K, img.shape[1], img.shape[0], eulers[0], eulers[1], eulers[2], gradient_clip)
    out = cv2.warpPerspective(img, H, bounds)
    if mask is None:
        mask = np.ones(img.shape[:2], dtype='uint8')
    mask = cv2.warpPerspective(mask, H, bounds)
    pts = [k.pt for k in keypoints]
    pts = H @ np.concatenate((np.array(pts), np.ones((len(pts), 1))), axis=1).T
    pts = pts[:2, :].T
    for k, pt in zip(keypoints, pts):
        k.pt = tuple(pt)
    return out, mask, keypoints


def scale_img(img: np.ndarray, keypoints: list, scale: float):
    if scale < 1:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    S = np.eye(3, dtype=float)
    S[0,0] = scale
    S[1,1] = scale
    pts = [k.pt for k in keypoints]
    pts = S @ np.concatenate((np.array(pts), np.ones((len(pts), 1))), axis=1).T
    pts = pts[:2, :].T
    for k, pt in zip(keypoints, pts):
        k.pt = tuple(pt)
    return img, keypoints


## euler rotation methods
# Matrix for Yaw-rotation about the Z-axis
def R_z(psi, degrees=False):
    psi = psi * np.pi / 180 if degrees else psi
    R = np.array([[np.cos(psi), -np.sin(psi), 0, 0],
                   [np.sin(psi), np.cos(psi), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return R


# Matrix for Pitch-rotation about the Y-axis
def R_y(theta, degrees=False):
    theta = theta * np.pi / 180 if degrees else theta
    R = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                   [0, 1, 0, 0],
                   [-np.sin(theta), 0, np.cos(theta), 0],
                   [0, 0, 0, 1]])
    return R


# Matrix for Roll-rotation about the X-axis
def R_x(phi, degrees=False):
    phi = phi*np.pi/180 if degrees else phi
    R = np.array([[1, 0, 0, 0],
                   [0, np.cos(phi), -np.sin(phi), 0],
                   [0, np.sin(phi), np.cos(phi), 0],
                   [0, 0, 0, 1]])
    return R


def euler_rotation(phi, theta, psi, degrees=False):
    return np.linalg.multi_dot([R_x(phi, degrees), R_y(theta, degrees), R_z(psi, degrees)])


def euler_affine_rotate(img: np.ndarray, euler: np.array, degrees: bool=False):
    """img: numpy array image
    euler: numpy array euler angles in Yaw, Pitch, Roll order"""
    if degrees:
        euler = np.deg2rad(euler)
    R = Rotation.from_euler("zyx", euler, degrees=False).as_matrix()
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


def euler_perspective_rotate(img: np.ndarray, euler: np.array, degrees: bool=False):
    """img: numpy array image
    euler: numpy array euler angles in Yaw, Pitch, Roll order"""
    if degrees:
        euler = np.deg2rad(euler)
    R = Rotation.from_euler("zyx", euler, degrees=False).as_matrix()
    height, width, channels = img.shape
    src = np.array([[0, 0, 0],
                    [0, height-1, 0],
                    [width-1, height-1, 0],
                    [width-1, 0, 0]], dtype="float32")
    # the destination points are rotated by R
    dst = np.matmul(src,R).astype("float32")
    src = src.astype("float32")
    rvec = np.zeros((3,1),dtype="float32")
    #rvec, _ = cv2.Rodrigues(R)
    tvec = np.zeros((3,1),dtype="float32")
    #tvec = np.array([height/2, width/2, 0], dtype="float32")
    A = np.eye(3, dtype="float32")
    d = np.zeros((5,1), dtype="float32")
    dst_projection, _ = cv2.projectPoints(dst, rvec, tvec, A, d)
    dst_projection = dst_projection.squeeze()
    src = src[:,0:2]
    pers_tform = cv2.getPerspectiveTransform(src.astype("float32"), dst_projection.astype("float32"))
    bx, by, bwidth, bheight = cv2.boundingRect(dst_projection)
    img_dst = cv2.warpPerspective(img, pers_tform, (bwidth, bheight))
    return img_dst

## Bird View method- get a top down view of the frames
def bird_view(image, pitch=45):
    ## Crop image

    IMAGE_H = image.shape[0]
    IMAGE_W = image.shape[1]

    # Assume that focus length is equal half image width
    FOCUS = IMAGE_W / 2  ## Focus needs to stay the same height of the seabed-camera

    # Translation of images on the black canvases
    translation = np.float32(([1, 0,  IMAGE_W], [0, 1, IMAGE_H], [0, 0, 1]))
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

    return warped_img


import numpy as np
import cv2


# Usage:
#     Change main function with ideal arguments
#     Then
#     from image_tranformer import ImageTransformer
#
# Parameters:
#     image_path: the path of image that you want rotated
#     shape     : the ideal shape of input image, None for original size.
#     theta     : rotation around the x axis
#     phi       : rotation around the y axis
#     gamma     : rotation around the z axis (basically a 2D rotation)
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis (distance to the image)
#
# Output:
#     image     : the rotated image
#
# Reference:
#     1.        : http://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
#     2.        : http://jepsonsblog.blogspot.tw/2012/11/rotation-in-3d-using-opencvs.html

class ImageTransformer(object):
    """ Perspective transformation class for image
        with shape (height, width, #channels) """

    def __init__(self, image: np.ndarray):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.num_channels = self.image.shape[2]

    """ Wrapper of Rotating a Image """

    def rotate_along_axis(self, theta: float=0, phi: float=0, gamma: float=0, dx: float=0, dy: float=0, dz: float=0, degrees: bool=False):
        # Get radius of rotation along 3 axes
        if degrees:
            rtheta = np.deg2rad(theta)
            rphi = np.deg2rad(phi)
            rgamma = np.deg2rad(gamma)
        else:
            rtheta = theta
            rphi = phi
            rgamma = gamma

        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height ** 2 + self.width ** 2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)

        corners = np.array([[[0, 0],
                             [self.width, 0],
                             [self.width, self.height],
                             [0, self.height]]], dtype="float32")

        dst_corners = cv2.perspectiveTransform(corners, mat)
        bx, by, bwidth, bheight = cv2.boundingRect(dst_corners)
        t = np.array([[1, 0, -bx],
                      [0, 1, -by],
                      [0, 0, 1]])

        mat = t.dot(mat)
        img = cv2.warpPerspective(self.image.copy(), mat, (bwidth, bheight))
        return img
        return cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height))

    """ Get Perspective Projection Matrix """

    def get_M(self, theta, phi, gamma, dx, dy, dz):
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([[1, 0, -w / 2],
                       [0, 1, -h / 2],
                       [0, 0, 1],
                       [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([[1, 0, 0, 0],
                       [0, np.cos(theta), -np.sin(theta), 0],
                       [0, np.sin(theta), np.cos(theta), 0],
                       [0, 0, 0, 1]])

        RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0],
                       [0, 1, 0, 0],
                       [np.sin(phi), 0, np.cos(phi), 0],
                       [0, 0, 0, 1]])

        RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                       [np.sin(gamma), np.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([[f, 0, w / 2, 0],
                       [0, f, h / 2, 0],
                       [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

