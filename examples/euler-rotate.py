from skimage import io
import numpy as np
from mosaicking.transformations import euler_affine_rotate, euler_perspective_rotate
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str, help="Path to the image file that needs transforming.")
parser.add_argument("-p", "--pitch", nargs='?', default=0., type=float, help="Static pitch angle to use.")
parser.add_argument("-r", "--roll", nargs='?', default=0., type=float, help="Static roll angle to use.")
parser.add_argument("-y", "--yaw", nargs='?', default=0., type=float, help="Static yaw angle to use.")
parser.add_argument("-d", "--degrees", action="store_true", help="flag to specify if angles are in degrees")
args = parser.parse_args()

input_path = Path(args.filepath).resolve()
Path("./outputs").resolve().mkdir(parents=True, exist_ok=True)
output_path = Path("./outputs").resolve().joinpath("affine-underwater-fishes.png")
img = io.imread(input_path)
euler = np.array([args.yaw, args.pitch, args.roll])
img_dst = euler_perspective_rotate(img, euler, True)
io.imshow(img_dst)
#io.imsave(output_path, img_dst)
