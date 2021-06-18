import numpy as np

from mosaicking.preprocessing import *
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
    parser.add_argument("-s", "--show", action="store_true", help="Show the output images.")
    parser.add_argument("-r", "--roi", type=int, nargs=4, default=None, help="Space delimited list of bbox to crop to, order is width, height, top left x, top left y")
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

    if args.roi is not None:
        width = args.roi[0]
        height = args.roi[1]

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


    R = Rotation.from_euler("xyz", [args.xrotation, args.yrotation, args.zrotation])
    H, (xmin,xmax), (ymin, ymax) = calculate_homography(K, width, height, R.as_matrix(), np.zeros(3), args.gradientclip)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (xmax-xmin, ymax-ymin))
    try:  # External try to handle any unexpected errors
        while not evaluate_stopping(reader, args):
            sys.stdout.write("Processing Frame {}\n".format(int(reader.get(cv2.CAP_PROP_POS_FRAMES) + 1)))
            # Acquire a frame
            ret, img = reader.read()
            if args.roi is not None:
                roi = [args.roi[-1], args.roi[-1]+args.roi[1], args.roi[-2], args.roi[-2]+args.roi[0]]
                img = img[roi[0]:roi[1], roi[2]:roi[3], :]
            # Fix color, lighting, contrast
            img = rebalance_color(img, 1, 0.5, 1)
            img = fix_light(img)
            img = fix_contrast(img)
            img = fix_color(img, 0.9)
            # Rotate the image
            out = cv2.warpPerspective(img, H, (xmax - xmin, ymax - ymin), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # write the warped frame
            writer.write(out)
            out = cv2.resize(out, (0,0),fx=0.3,fy=0.3,interpolation=cv2.INTER_AREA)
            if args.show:
                cv2.imshow("warp",out)
                cv2.imshow("img", cv2.resize(img, (0,0),fx=0.3,fy=0.3,interpolation=cv2.INTER_AREA))
            key = cv2.waitKey(1)
            if key > -1:
                break
    except:
        pass
    cv2.destroyAllWindows()
    reader.release()
    writer.release()

