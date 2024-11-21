import numpy as np
from scipy.spatial.transform import Rotation
import sys
from numpy import typing as npt
from typing import Sequence, Union
import cv2

import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


def calculate_homography(K: npt.NDArray[float],
                         width: int, height: int,
                         R: Union[Rotation | npt.NDArray[float]], T: npt.NDArray[float],
                         gradient_clip: float=0.0) -> tuple[npt.NDArray[float], tuple[int, int], tuple[int, int]]:
    """
    Calculates the homography matrix from a set of points. Applies an optional gradient clipping to the homography matrix
     (cropping).
    :param K: Camera intrinsic matrix (3x3)
    :type K: npt.NDArray[float]
    :param width: Image width (pix)
    :type width: int
    :param height: Image height (pix)
    :type height: int
    :param R: Camera rotation matrix (3x3)
    :type R: Union[Rotation | npt.NDArray[float]]
    :param T: Camera translation vector (3x1)
    :type T: npt.NDArray[float]
    :param gradient_clip: Warping gradient threshold to allow, components of the image greater than this threshold are
    cropped..
    :type gradient_clip: float
    :returns:
        - **H** (*npt.NDArray[float]*): Homography matrix
        - **x_bounds** (*tuple[float]*): Minimum and maximum x bounds of the output image.
        - **y_bounds** (*tuple[float]*): Minimum and maximum y bounds of the output image.
    :rtype: tuple(npt.NDArray[float], tuple[float], tuple[float])
    """
    if K.shape[1] < 4:
        K = np.concatenate((K, np.zeros((3, 1))), axis=1)
    # Calculate the inverse of K
    Kinv = np.zeros((4, 3))
    Kinv[:3, :3] = np.linalg.inv(K[:3, :3]) * (K[0, 0] * K[1, 1])
    Kinv[-1, :] = [0, 0, 1]
    # Get the extrinsic matrix (Camera to World)
    E = get_extrinsic_matrix(R.as_matrix(), T) if isinstance(R, Rotation) else get_extrinsic_matrix(R, T)
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
            logger.warning("Gradient Explosion. Is the Image Rapidly Zooming In/Out or pitching / rolling a lot? Gradient clipping is suppressed to allow continuity, consider increasing gradient clip argument.")
        else:
            pts = pts[:, idx]
    # Round inwards to pixel centers
    xmin, ymin = np.int32(pts.min(axis=1) + 0.5)
    xmax, ymax = np.int32(pts.max(axis=1) - 0.5)
    T1 = np.array([[1, 0, -xmin],
                   [0, 1, -ymin],
                   [0, 0, 1]])
    H = np.linalg.multi_dot([T1, K, E, Kinv])
    return H, (xmin, xmax), (ymin, ymax)


def apply_transform(img: npt.NDArray[np.uint8],
                    K: npt.NDArray[float],
                    R: Union[Rotation | npt.NDArray[float]],
                    T: npt.NDArray[float], keypoints: Sequence[cv2.KeyPoint],
                    scale: float = None, mask: npt.NDArray[np.uint8] = None,
                    gradient_clip: float = 0.0) -> tuple[npt.NDArray[np.uint8], Sequence[cv2.KeyPoint], npt.NDArray[np.uint8]]:
    """
    Warps an image and feature keypoints by composing a perspective homography matrix based on a provide camera intrinsic matrix and extrinsic matrix.
    :param img: Input image
    :type img: npt.NDArray[np.uint8]
    :param K: Camera intrinsic matrix (3x3)
    :type K: npt.NDArray[float]
    :param R: Camera rotation matrix (3x3)
    :type R: Union[Rotation | npt.NDArray[float]]
    :param T: Camera translation vector (3x1)
    :type T: npt.NDArray[float]
    :param keypoints: Set of keypoint features to transform.
    :type keypoints: Sequence[cv2.KeyPoint]
    :param scale: zoom scale > 1 upsizes image, zoom scale < 1 downsizes image
    :type scale: float
    :param mask: Mask to apply to image
    :type mask: npt.NDArray[np.uint8]
    :param gradient_clip: Warping gradient threshold to allow, components of the image greater than this threshold are
    cropped.
    :type gradient_clip: float
    :returns:
        - **warped_image** (*npt.NDArray[np.uint8]*): Warped image
        - **warped_keypoints** (*Sequence[cv2.KeyPoint]*): Warped keypoints
        - **warped_mask** (*npt.NDArray[np.uint8]*): Warped mask
    :rtype: tuple(npt.NDArray[np.uint8], Sequence[cv2.KeyPoint], npt.NDArray[np.uint8])
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
    if scale is not None:
        img_warped, kp, mask_warped = apply_scale(img_warped, kp, scale, mask_warped)
    return img_warped, kp, mask_warped


def apply_scale(img: npt.NDArray[np.uint8], keypoints: Sequence[cv2.KeyPoint], scale: float, mask: npt.NDArray[np.uint8] = None) -> tuple[npt.NDArray[np.uint8], Sequence[cv2.KeyPoint], npt.NDArray[np.uint8]]:
    """
    Scales an image and feature keypoints by a constant scaling factor.
    :param img: Input image
    :type img: npt.NDArray[np.uint8]
    :param keypoints: Set of keypoint features to transform.
    :type keypoints: Sequence[cv2.KeyPoint]
    :param scale: zoom scale > 1 upsizes image, zoom scale < 1 downsizes image
    :type scale: float
    :param mask: Mask to apply to image
    :type mask: npt.NDArray[np.uint8]
    :returns:
        - **scaled_image** (*npt.NDArray[np.uint8]*): Scaled image
        - **scaled_keypoints** (*Sequence[cv2.KeyPoint]*): Scaled keypoints
        - **scaled_mask** (*npt.NDArray[np.uint8]*): Scaled mask
    :rtype: tuple(npt.NDArray[np.uint8], Sequence[cv2.KeyPoint], npt.NDArray[np.uint8])
    """
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
    return img, kp, mask


def get_extrinsic_matrix(Rc: Union[Rotation | npt.NDArray[float]],
                         C: npt.NDArray[float], order: str = "xyz",
                         degrees: bool=False) -> npt.NDArray[float]:
    """
    Computes the extrinsic matrix for a camera given its rotation and position in world coordinates.

    :param Rc: Rotation matrix or a Rotation object representing the orientation of the camera.
    :type Rc: Union[Rotation, npt.NDArray[float]]
    :param C: Camera position in world coordinates.
    :type C: npt.NDArray[float]
    :param order: Order of rotations applied (default: "xyz").
    :type order: str
    :param degrees: Whether the rotation angles in `Rc` are specified in degrees (default: False).
    :type degrees: bool
    :returns: **extrinsic_matrix** (*npt.NDArray[float]*): 4x4 transformation matrix that maps world coordinates to camera coordinates.
    :rtype: npt.NDArray[float]
    """
    if isinstance(Rc, np.ndarray):
        if Rc.size not in [3, 4, 9]:
            raise ValueError("Rotation must be of size 3, 4 or 9.")
        if C.size != 3:
            raise ValueError("Translation must be of size 3.")
        C = C.reshape((3, 1))
        if Rc.size == 3:
            Rc = Rotation.from_euler(order, Rc, degrees=degrees).as_matrix()
        elif Rc.size == 4:
            Rc = Rotation.from_quat(Rc).as_matrix()
        else:
            if Rc.ndim < 2:
                Rc = Rc.reshape((3, 3))
            Rc = Rotation.from_matrix(Rc).as_matrix()
    else:
        Rc = Rc.as_matrix()
    R = Rc.T
    t = -R @ C
    return np.concatenate((np.concatenate((R, t), axis=1), [[0,0,0,1]]), axis=0)


## euler rotation methods
# Matrix for Yaw-rotation about the Z-axis
def R_z(psi: float, degrees=False) -> npt.NDArray[float]:
    """
    Creates a 2D rotation matrix for a rotation around the Z-axis.

    :param psi: The rotation angle.
    :type psi: float
    :param degrees: Whether the angle is specified in degrees (default: False).
    :type degrees: bool
    :returns: 3x3 rotation matrix for Z-axis rotation.
    :rtype: npt.NDArray[float]
    """
    psi = psi * np.pi / 180 if degrees else psi
    R = np.array([[np.cos(psi), -np.sin(psi), 0, 0],
                   [np.sin(psi), np.cos(psi), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return R


# Matrix for Pitch-rotation about the Y-axis
def R_y(theta: float, degrees=False) -> npt.NDArray[float]:
    """
    Creates a 2D rotation matrix for a rotation around the Y-axis.

    :param theta: The rotation angle.
    :type theta: float
    :param degrees: Whether the angle is specified in degrees (default: False).
    :type degrees: bool
    :returns: 3x3 rotation matrix for Y-axis rotation.
    :rtype: npt.NDArray[float]
    """
    theta = theta * np.pi / 180 if degrees else theta
    R = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                   [0, 1, 0, 0],
                   [-np.sin(theta), 0, np.cos(theta), 0],
                   [0, 0, 0, 1]])
    return R


# Matrix for Roll-rotation about the X-axis
def R_x(phi: float, degrees=False) -> npt.NDArray[float]:
    """
    Creates a 2D rotation matrix for a rotation around the X-axis.

    :param phi: The rotation angle.
    :type phi: float
    :param degrees: Whether the angle is specified in degrees (default: False).
    :type degrees: bool
    :returns: 3x3 rotation matrix for X-axis rotation.
    :rtype: npt.NDArray[float]
    """
    phi = phi*np.pi/180 if degrees else phi
    R = np.array([[1, 0, 0, 0],
                   [0, np.cos(phi), -np.sin(phi), 0],
                   [0, np.sin(phi), np.cos(phi), 0],
                   [0, 0, 0, 1]])
    return R


def euler_rotation(phi: float, theta: float, psi: float, degrees=False) -> npt.NDArray[float]:
    """
    Creates a rotation matrix from Euler angles for rotations around the X, Y, and Z axes.

    :param phi: Rotation around the X-axis.
    :type phi: float
    :param theta: Rotation around the Y-axis.
    :type theta: float
    :param psi: Rotation around the Z-axis.
    :type psi: float
    :param degrees: Whether the angles are specified in degrees (default: False).
    :type degrees: bool
    :returns: 3x3 rotation matrix from Euler angles.
    :rtype: npt.NDArray[float]
    """
    return np.linalg.multi_dot([R_x(phi, degrees), R_y(theta, degrees), R_z(psi, degrees)])


def euler_affine_rotate(img: npt.NDArray[np.uint8], euler: npt.NDArray[float], degrees: bool=False) -> npt.NDArray[np.uint8]:
    """
    Applies an affine rotation to an image based on Euler angles.

    :param img: Input image.
    :type img: npt.NDArray[np.uint8]
    :param euler: Array of Euler angles for X, Y, and Z rotations.
    :type euler: npt.NDArray[float]
    :param degrees: Whether the Euler angles are specified in degrees (default: False).
    :type degrees: bool
    :returns: Rotated image.
    :rtype: npt.NDArray[np.uint8]
    """
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


def euler_perspective_rotate(img: npt.NDArray[np.uint8], euler: npt.NDArray[float], degrees: bool=False) -> npt.NDArray[np.uint8]:
    """
    Applies a perspective rotation to an image based on Euler angles.

    :param img: Input image.
    :type img: npt.NDArray[np.uint8]
    :param euler: Array of Euler angles for X, Y, and Z rotations.
    :type euler: npt.NDArray[float]
    :param degrees: Whether the Euler angles are specified in degrees (default: False).
    :type degrees: bool
    :returns: Rotated image.
    :rtype: npt.NDArray[np.uint8]
    """
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
def bird_view(image: npt.NDArray[np.uint8], pitch: float = 45) -> npt.NDArray[np.uint8]:
    """
    Applies a bird's-eye (top-down) view transformation to an image.

    :param image: Input image.
    :type image: npt.NDArray[np.uint8]
    :param pitch: Pitch angle for the transformation (default: 45).
    :type pitch: float
    :returns: Image transformed to a bird's-eye view.
    :rtype: npt.NDArray[np.uint8]
    """
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


def get_origin_direction(E: npt.NDArray[float]) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Extracts the camera's origin and direction vector from a 4x4 extrinsic matrix.

    :param E: 4x4 extrinsic matrix.
    :type E: npt.NDArray[float]
    :returns:
        - **origin** (*npt.NDArray[float]*): Camera origin in world coordinates.
        - **direction** (*npt.NDArray[float]*): Direction vector pointing towards the scene.
    :rtype: tuple(npt.NDArray[float], npt.NDArray[float])
    """

    # Extract the rotation matrix and translation vector
    R = E[:3, :3]
    t = E[:3, 3]

    # The camera's position in world coordinates is the negative translation vector
    origin = -R.T @ t

    # The camera's direction is the third column of the rotation matrix
    direction = R[:, 2]

    return origin, direction


def get_alignment(src_pts: npt.NDArray[float], src_shape: tuple[int, int], dst_pts: npt.NDArray[float], dst_shape: tuple[int, int], homography: str = "similar", gradient: float = 0.0) -> tuple[npt.NDArray[float], tuple[int, int], tuple[int, int]]:
    """
    Calculates the alignment transformation between two sets of points with optional homography.

    :param src_pts: Source points for the alignment.
    :type src_pts: npt.NDArray[float]
    :param src_shape: Shape (height, width) of the source image.
    :type src_shape: tuple[int, int]
    :param dst_pts: Destination points for the alignment.
    :type dst_pts: npt.NDArray[float]
    :param dst_shape: Shape (height, width) of the destination image.
    :type dst_shape: tuple[int, int]
    :param homography: Type of homography to apply ("similar", "affine", etc.) (default: "similar").
    :type homography: str
    :param gradient: Gradient parameter for the alignment (default: 0.0).
    :type gradient: float
    :returns:
        - **transform** (*npt.NDArray[float]*): Transformation matrix.
        - **aligned_src_shape** (*tuple[int, int]*): Aligned source image shape.
        - **aligned_dst_shape** (*tuple[int, int]*): Aligned destination image shape.
    :rtype: tuple(npt.NDArray[float], tuple[int, int], tuple[int, int])
    """
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
    elif homography == "affine":
        A, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1)
    elif homography == "perspective":
        A, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1)
    else:
        raise ValueError(f'homography argument not supported, choose from {["rigid", "similar", "affine", "perspective"]}')
    A = np.concatenate((A, np.array([[0, 0, 1]])), axis=0) if A.size < 9 else A  # convert to a homogeneous form

    # Warp the image coordinates
    u_idx, v_idx = np.indices(tuple(s - 1 for s in src_shape[1::-1]), float)
    grid_dst = np.stack((u_idx, v_idx, np.ones_like(u_idx)), axis=2).reshape((-1, 1, 3))
    warp_grid_dst = A @ grid_dst.squeeze().T
    warp_grid_dst = warp_grid_dst[:2, :] / warp_grid_dst[-1, :]

    if gradient > 0:
        # Compute the gradient of the transformed coordinates
        grad_x = cv2.Sobel(warp_grid_dst.T[..., 0], cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(warp_grid_dst.T[..., 1], cv2.CV_64F, 0, 1, ksize=5)
        # Compute the norm of the gradient
        grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
        idx = (grad_norm < gradient).squeeze()
        if not idx.any():
            sys.stderr.write("WARNING: Gradient Explosion. Is the Image Rapidly Zooming In/Out? Gradient clipping is suppressed to allow continuity, consider increasing gradient clip argument.\n")
        else:
            warp_grid_dst = warp_grid_dst[:, idx]

    # Get the corners of the mosaic image in homogeneous coords (x,y,w=1)
    dst_crn = np.array([[0, dst_shape[1], dst_shape[1], 0],
                        [0, 0, dst_shape[0], dst_shape[0]]], float)

    # Concatenate the mosaic and warped corner coordinates
    pts = np.concatenate([dst_crn, warp_grid_dst], axis=1)

    # Round to pixel centers
    xmin, ymin = np.int32(pts.min(axis=1) + 0.5)
    xmax, ymax = np.int32(pts.max(axis=1) + 0.5)

    t = [-xmin, -ymin]  # C of the upper left corner of the transformed image
    A[:2, -1] = A[:2, -1] + t  # C homography
    return A, (xmin, xmax), (ymin, ymax)


def remove_z_rotation(rotation: Rotation) -> Rotation:
    """
    Removes the Z (yaw) component of a rotation, retaining only the X and Y components.

    :param rotation: A Rotation object.
    :type rotation: Rotation
    :returns: A new Rotation object with the Z rotation component removed.
    :rtype: Rotation
    """
    # Decompose the original rotation into Euler angles (yaw, pitch, roll)
    # 'zyx' means the input rotation is first around z, then y, then x
    ypr = rotation.as_euler('zyx')

    # Set yaw to zero to remove rotation about the Z-axis
    if ypr.ndim > 1:
        ypr[:, 0] = 0
    else:
        ypr[0] = 0

    # Create a new rotation using the modified yaw and original pitch and roll
    # Note that the angles must be provided in the reverse order of axes
    new_rotation = Rotation.from_euler('zyx', ypr)

    return new_rotation


def homogeneous_translation(x: float, y: float) -> np.ndarray:
    out = np.eye(3)
    out[[0, 1], 2] = [x, y]
    return out


def homogeneous_scaling(*sf: Sequence[float]) -> np.ndarray:
    assert len(sf) < 2, "Maximum two scaling factors allowed."
    out = np.eye(3)
    out[[0, 1], [0, 1]] = sf
    return out


def inverse_K(K: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of a zero-skew pinhole-camera projection matrix.
    """
    K_inv = np.eye(*K.shape)
    K_inv[0, 0] = K[1, 1]
    K_inv[1, 1] = K[0, 0]
    K_inv[0, 1] = -K[0, 1]
    K_inv[0, 2] = K[1, 2] * K[0, 1] - K[0, 2] * K[1, 1]
    K_inv[1, 2] = -K[1, 2] * K[0, 0]
    K_inv[2, 2] = K[0, 0] * K[1, 1]
    return 1 / (K[0, 0] * K[1, 1]) * K_inv
