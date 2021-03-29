import cv2
import numpy as np
from scipy.spatial import transform
from matplotlib import pyplot as plt

fname = 'images/registration/scene1.png'
img = cv2.imread(fname, cv2.IMREAD_COLOR)
euler = np.array([[45],[45],[30]], dtype="float32")

# First take the corners of the image and project into a pseudo 3D space where depth is half the image width
height,width,channels = img.shape

# X is a 3x4 matrix, each column is a point corresponding to Top Left, Top Right, Bottom Right, Bottom Left cornerrs of image
X = np.array([[0, 0, width/2],
              [width, 0, width/2],
              [width, height, width/2],
              [0, height, width/2]], dtype="float32").T

# Compose the rotation matrix out of euler
R = transform.Rotation.from_euler("xyz", euler.T, True)

X1 = R.as_matrix().squeeze() @ X

H, _ = cv2.findHomography(X[:2,:].T, X1[:2,:].T, cv2.RANSAC)

xmin,ymin,_ = np.int32(X1.min(axis=1) - 0.5)
xmax,ymax,_ = np.int32(X1.max(axis=1) + 0.5)

# The translation operator for tile
t = [-xmin, -ymin]
Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translation homography

out = cv2.warpPerspective(img, Ht.dot(H), (xmax - xmin, ymax - ymin))

fig,ax = plt.subplots()
ax.imshow(img)
fig,ax = plt.subplots()
ax.imshow(out)
plt.show()