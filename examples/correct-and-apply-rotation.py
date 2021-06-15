import numpy as np

from mosaicking.preprocessing import fix_color, fix_contrast, fix_light
from mosaicking.utils import *
from mosaicking.transformations import *
from pathlib import Path
import argparse
import cv2
import sys


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to video file.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start_time", type=float, help="Time (secs) to start from.")
    group.add_argument("--start_frame", type=int, help="Frame number to start from.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--finish_time", type=float, help="Time (secs) to finish at.")
    group.add_argument("--finish_frame", type=int, help="Frame number to finish at.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c","--calibration", type=str, default=None, help="Path to calibration file.")
    group.add_argument("-k","--intrinsic",nargs=9, type=float, default=None, help="Space delimited list of intrinsic matrix terms, Read as K[0,0],K[1,0],K[2,0],K[1,0],K[1,1],K[1,2],K[2,0],K[2,1],K[2,2]")
    parser.add_argument("-x","--xrotation", type=float, default=0, help="Rotation around image plane's x axis (radians).")
    parser.add_argument("-y", "--yrotation", type=float, default=0, help="Rotation around image plane's y axis (radians).")
    parser.add_argument("-z", "--zrotation", type=float, default=0, help="Rotation around image plane's z axis (radians).")
    parser.add_argument("-g", "--gradientclip", type=float, default=0, help="Clip the gradient of severely distorted image.")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    assert video_path.exists(), "File not found: {}".format(str(video_path))
    video_out = video_path.parent.joinpath("Rotated_{}".format(video_path.name))

    reader = cv2.VideoCapture(str(video_path))
    reader = get_starting_pos(reader, args)
    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.intrinsic is not None:
        K = np.array(args.intrinsic).reshape((3,3)).T
    elif args.calibration is not None:
        K = np.eye(3)
        K[0,2] = float(width)/2
        K[1,2] = float(height)/2
    else:
        K = np.eye(3)
        K[0, 2] = float(width) / 2
        K[1, 2] = float(height) / 2
    print("K: {}".format(repr(K)))

    if K.shape[1] < 4:
        K = np.concatenate((K, np.zeros((3, 1))), axis=1)
    Kinv = np.zeros((4, 3))
    Kinv[:3, :3] = np.linalg.inv(K[:3, :3]) * (K[0, 0] * K[1, 1])
    Kinv[-1, :] = [0, 0, 1]
    R = euler_rotation(args.xrotation, args.yrotation, args.zrotation)
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
    if args.gradientclip > 0:
        grad = np.gradient(pts, axis=1)
        idx = np.sqrt((grad ** 2).sum(axis=0)) < args.gradientclip
        pts = pts[:, idx]
    # Round to pixel centers
    xmin, ymin = np.int32(pts.min(axis=1) - 0.5)
    xmax, ymax = np.int32(pts.max(axis=1) + 0.5)
    T1 = np.array([[1, 0, -xmin],
                   [0, 1, -ymin],
                   [0, 0, 1]])
    H = np.linalg.multi_dot([T1, K, R, T, Kinv])


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (xmax-xmin, ymax-ymin))
    try:  # External try to handle any unexpected errors
        while not evaluate_stopping(reader, args):
            sys.stdout.write("Processing Frame {}\n".format(int(reader.get(cv2.CAP_PROP_POS_FRAMES) + 1)))
            # Acquire a frame
            ret, img = reader.read()
            # Fix color, lighting, contrast
            #

            img = (img.astype(float)*[1.5,0.5,0.8]).astype(np.uint8)
            img = fix_light(img)

            img = fix_contrast(img)
            img = fix_color(img, 0.9)
            # Rotate the image
            out = cv2.warpPerspective(img, H, (xmax - xmin, ymax - ymin), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # write the warped frame
            writer.write(out)
            out = cv2.resize(out, (0,0),fx=0.3,fy=0.3,interpolation=cv2.INTER_AREA)
            cv2.imshow("warp",out)
            key = cv2.waitKey(5)
            if key == ord("x"):
                break
    except:
        pass
    cv2.destroyAllWindows()
    reader.release()
    writer.release()

