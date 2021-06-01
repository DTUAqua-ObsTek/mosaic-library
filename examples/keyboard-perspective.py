import keyboard
import numpy as np
from mosaicking.transformations import ImageTransformer
import cv2


img = cv2.imread("../images/corrections/underwater-fishes.png")
width,height,channels = img.shape
tformer = ImageTransformer(img)
warped = img
cv2.namedWindow("W/S to tilt around X axis. A/D to tilt around Y axis. Q/E to tilt around Z axis.", cv2.WINDOW_NORMAL)
cv2.imshow("W/S to tilt around X axis. A/D to tilt around Y axis. Q/E to tilt around Z axis.", warped)
cv2.waitKey(1)
euler = np.zeros((3,),dtype="float32")
scaler = 1.0
while True:
    key = keyboard.read_event()
    if key.name+key.event_type == "wdown":
        euler[1]+=scaler
    elif key.name+key.event_type == "sdown":
        euler[1]-=scaler
    elif key.name + key.event_type == "adown":
        euler[2]+=scaler
    elif key.name + key.event_type == "ddown":
        euler[2]-=scaler
    elif key.name + key.event_type == "qdown":
        euler[0]+=scaler
    elif key.name + key.event_type == "edown":
        euler[0]-=scaler
    elif key.name + key.event_type == "page updown":
        scaler+=1
    elif key.name + key.event_type == "pgdndown":
        scaler-=1
    elif key.name + key.event_type == "escdown":
        break
    elif key.name + key.event_type == "spacedown":
        euler = np.zeros((3,), dtype="float32")
        scaler = 1.0
    else:
        print(key.name + key.event_type)

    warped = tformer.rotate_along_axis(euler[1], euler[2], euler[0], degrees=True)
    cv2.imshow("W/S to tilt around X axis. A/D to tilt around Y axis. Q/E to tilt around Z axis.", warped)
    cv2.waitKey(1)
cv2.destroyAllWindows()
