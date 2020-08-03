from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from mosaicking.transformations import euler_affine_rotate
from pathlib import Path

input_path = Path("../images/corrections/underwater-fishes.png").resolve()
output_path = Path("../images/outputs").resolve().joinpath("affine-underwater-fishes.png")
img = io.imread(input_path)
euler = np.array([45, 25, 60])
img_dst = euler_affine_rotate(img, euler, True)
io.imsave(output_path, img_dst)
