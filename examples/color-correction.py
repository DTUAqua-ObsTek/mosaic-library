from skimage import io
from mosaicking.preprocessing import fix_color, fix_contrast, fix_light
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str, help="Path to the image file that needs correcting.")
args = parser.parse_args()

input_path = Path(args.directory).resolve()
img = io.imread(input_path)
img1 = fix_color(img)
img2 = fix_light(img1)
img3 = fix_contrast(img2)

io.imshow_collection([img, img1, img2, img3])

output_path = Path("./outputs").resolve()
output_path.mkdir(parents=True, exist_ok=True)
io.imsave(output_path.joinpath("color-corrected.png"), img1)
io.imsave(output_path.joinpath("contrast-corrected.png"), img3)
io.imsave(output_path.joinpath("lighting-corrected.png"), img2)