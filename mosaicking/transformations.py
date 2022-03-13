import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import sys
import copy


def calculate_homography(K: np.ndarray, width: int, height: int, R: np.ndarray, T: np.ndarray, gradient_clip: float=0.0):
    if K.shape[1] < 4:
        K = np.concatenate((K, np.zeros((3, 1))), axis=1)
    # Calculate the inverse of K
    Kinv = np.zeros((4, 3))
    Kinv[:3, :3] = np.linalg.inv(K[:3, :3]) * (K[0, 0] * K[1, 1])
    Kinv[-1, :] = [0, 0, 1]
    # Get the extrinsic matrix (Camera to World)
    E = get_extrinsic_matrix(R, T)
    #E = np.concatenate((np.concatenate((R, T.reshape(3,1)), axis=1), [[0,0,0,1]]), axis=0)
    # The homography is a set of transformations, first transform the image frame onto the camera frame
    # Then Rotate and translate the camera frame onto the world frame
    # Then transform back to the image frame
    H = np.linalg.multi_dot([K, E, Kinv])
    # Warp a grid of points on the image plane
    xgrid = np.arange(0, width)
    ygrid = np.arange(0, height)
    xx, yy = np.meshgrid(xgrid, ygrid, indexing='ij')
    grid = np.stack((xx.flatten(), yy.flatten(), np.ones_like(yy.flatten())), 0)
    warp_grid = H @ grid
    pts = warp_grid[:2, :] / warp_grid[-1, :]
    if gradient_clip > 0:
        warp_xx = pts[0, :].reshape(xx.shape)
        warp_yy = pts[1, :].reshape(yy.shape)
        dGu = np.stack(np.gradient(warp_xx),axis=-1)
        dGv = np.stack(np.gradient(warp_yy),axis=-1)
        slope = np.sqrt((dGu**2).sum(axis=-1) + (dGv**2).sum(axis=-1))
        safe = slope < gradient_clip
        xindx = np.argwhere(safe.any(axis=1))  # columns where safe exists
        yindx = np.argwhere(safe.any(axis=0))  # rows where safe exists
        ii, jj = np.meshgrid(xindx, yindx)
        idx = np.ravel_multi_index((ii.flatten(),jj.flatten()), (xx.shape))
        if not idx.any():
            sys.stderr.write("WARNING: Gradient Explosion. Is the Image Rapidly Zooming In/Out? Gradient clipping is suppressed to allow continuity, consider increasing gradient clip argument.")
        else:
            pts = pts[:, idx]
    # Round inswards to pixel centers
    xmin, ymin = np.int32(pts.min(axis=1) + 0.5)
    xmax, ymax = np.int32(pts.max(axis=1) - 0.5)
    T1 = np.array([[1, 0, -xmin],
                   [0, 1, -ymin],
                   [0, 0, 1]])
    H = np.linalg.multi_dot([T1, K, E, Kinv])
    return H, (xmin, xmax), (ymin, ymax)


def apply_transform(img: np.ndarray, K: np.ndarray, R: Rotation, T: np.ndarray, keypoints: list, scale: float = None, mask: np.ndarray = None, gradient_clip: float = 0.0):
    """
    img: input image
    K: camera calibration matrix
    R: Rotation matrix from world to camera
    T: Translation vector (m) from world to camera
    keypoints: list of keypoints
    scale: zoom scaling
    mask: input mask
    gradient_clip: clip off warped components where spatial gradient is more than cut-off
    """
    H, (xmin,xmax), (ymin, ymax) = calculate_homography(K, img.shape[1], img.shape[0], R.as_matrix(), T, gradient_clip)
    img_warped = cv2.warpPerspective(img, H, (xmax - xmin, ymax - ymin), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if mask is None:
        mask = 255 * np.ones_like(img)[:, :, 0]
    mask_warped = cv2.warpPerspective(mask, H, (xmax - xmin, ymax - ymin), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Apply H to the keypoints
    pts = [k.pt for k in keypoints]
    pts = np.concatenate((np.array(pts), np.ones((len(pts), 1))), axis=1).T
    pts = H @ pts
    pts = (pts[:2, :]/pts[-1,:])
    pts = pts[:2, :].T
    kp = []
    for k, pt in zip(keypoints, pts):
        kp.append(cv2.KeyPoint(pt[0], pt[1], k.size, k.angle, k.response, k.octave, k.class_id))
    # for k, pt in zip(keypoints, pts):
    #     k.pt = tuple(pt)
    if scale is not None:
        img_warped, keypoints, mask_warped = apply_scale(img_warped, keypoints, scale, mask_warped)
    return img_warped, mask_warped, kp


def apply_scale(img: np.ndarray, keypoints: list, scale: float, mask: np.ndarray=None):
    if scale == 1.0:
        return img, keypoints, mask
    if scale < 1:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if mask is not None:
            mask = cv2.resize(mask, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    S = np.eye(3, dtype=float)
    S[0,0] = scale
    S[1,1] = scale
    pts = [k.pt for k in keypoints]
    pts = S @ np.concatenate((np.array(pts), np.ones((len(pts), 1))), axis=1).T
    pts = (pts[:2, :] / pts[-1, :]).T
    kp = []
    for k, pt in zip(keypoints, pts):
        kp.append(cv2.KeyPoint(pt[0], pt[1], k.size, k.angle, k.response, k.octave, k.class_id))
    # for k, pt in zip(keypoints, pts):
    #     k.pt = tuple(pt)
    return img, kp, mask


def get_extrinsic_matrix(Rc: np.ndarray, C: np.ndarray, order: str = "xyz", degrees: bool=False):
    """
    Converts a rotation from world to camera and C from world to camera into the correct extrinsic form (camera to world).
    @param Rc: Rotation from world frame to camera frame, can be a size 3 euler angle vector, quaternion (wxyz) or rotation matrix.
    @param C: Translation from world frame to camera frame, must be a size 3 vector.
    @param order: String specifying order for euler angles, must be a permutation of "xyz".
    @param degrees: Flag to indicate if euler angles are in degrees.
    """
    if Rc.size not in [3, 4, 9]:
        raise ValueError("Rotation must be of size 3, 4 or 9.")
    if C.size != 3:
        raise ValueError("Translation must be of size 3.")
    C = C.reshape((3, 1))
    if Rc.size == 3:
        Rc = Rotation.from_euler("xyz", Rc, degrees=degrees).as_matrix()
    elif Rc.size == 4:
        Rc = Rotation.from_quat(Rc).as_matrix()
    else:
        if Rc.ndim < 2:
            Rc = Rc.reshape((3, 3))
        Rc = Rotation.from_matrix(Rc).as_matrix()
    R = Rc.T
    t = -R @ C
    return np.concatenate((np.concatenate((R, t), axis=1), [[0,0,0,1]]), axis=0)


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
    fullTransformation = np.dot(translation, H)  # compose warp and C in correct order
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
#     dx        : C along the x axis
#     dy        : C along the y axis
#     dz        : C along the z axis (distance to the image)
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

