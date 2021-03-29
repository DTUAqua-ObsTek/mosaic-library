import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = "./images/registration/scene1.png"
img2 = "./images/registration/scene2.png"

prev_img = cv2.imread(img1, cv2.IMREAD_COLOR)
img = cv2.imread(img2, cv2.IMREAD_COLOR)

mapper = cv2.reg_MapperGradProj()
mapp_pyr = cv2.reg_MapperPyramid(mapper)
map_ptr = mapp_pyr.calculate(prev_img, img)
map_proj = cv2.reg.MapTypeCaster_toProjec(map_ptr)

#map_proj.normalize()

dest = map_proj.inverseWarp(img)

plt.imshow(np.concatenate((prev_img, img, dest), axis=1))