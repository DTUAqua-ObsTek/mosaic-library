from skimage import io
from mosaicking.preprocessing import fix_color, fix_contrast, fix_light
from pathlib import Path

input_path = Path("../images/corrections/underwater-fishes.png").resolve()
img = io.imread(input_path)
img1 = fix_color(img)
img2 = fix_light(img1)
img3 = fix_contrast(img2)

io.imshow_collection([img, img1, img2, img3])

output_path = Path("../images/outputs").resolve()
io.imsave(output_path.joinpath("color-corrected.png"), img1)
io.imsave(output_path.joinpath("contrast-corrected.png"), img3)
io.imsave(output_path.joinpath("lighting-corrected.png"), img2)