from skimage import io
from mosaicking.preprocessing import fix_color, fix_contrast, fix_light
from pathlib import Path
import numpy as np
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str, help="Path to the image file that needs correcting.")
args = parser.parse_args()

input_path = Path(args.filepath).resolve()
img = io.imread(input_path)
img1 = fix_color(img, percent=0.9)
img2 = fix_light(img1, limit=5, grid=(5,5), gray=False)
img3 = fix_contrast(img2)

cv2.namedWindow("Color Changes (Press any key to exit)", cv2.WINDOW_NORMAL)
cv2.imshow("Color Changes (Press any key to exit)", np.concatenate([img,img1,img2,img3], axis=1)[:,:,[2,1,0]] )
cv2.waitKey()
cv2.destroyAllWindows()
