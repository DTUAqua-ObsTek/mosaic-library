import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from mosaicking import preprocessing, registration, transformations
from skimage import io


########### Keep Window ration (BO) ###############
def showImageKeepRatio(winName, image, magnification=1.):
    # Shows an image sized with the factor magnification: default is 1
    if cv2.getWindowProperty(winName, cv2.WND_PROP_VISIBLE) == 0.0:
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winName, int(image.shape[1] * magnification), int(image.shape[0] * magnification))
    cv2.imshow(winName, image)


def createMosaic(args):
    # Add Video Directory
    videopath = Path(args.directory).resolve()
    if videopath.is_dir():
        raise Exception("No video chosen!")
    if args.output:
        outputpath = Path(args.output)
    else:
        outputpath = videopath.parent.joinpath("output_mosaic.png")

    # Start time
    start = time.time()
    keep_processing = True
    #    count = 0
    detector = cv2.xfeatures2d.SIFT_create(1000, 3)

    # define video capture object
    cap = cv2.VideoCapture(str(videopath))
    video_out = outputpath.parent.joinpath("output_mosaic.avi")
    # writer = cv2.VideoWriter(video_out, fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=30.0, frameSize=(640, 240), isColor=True)
    frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total Frame Number: {:d}'.format(int(frame_number)))

    ## Read first frame
    ret, in_frame = cap.read()

    # Resize
    in_frame = preprocessing.const_ar_scale(in_frame, 0.6)

    # Fix lighting, colors, contrast here
    in_frame = preprocessing.fix_color(in_frame, 1.0)
    in_frame = preprocessing.fix_light(in_frame)
    in_frame = preprocessing.fix_contrast(in_frame)

    # Apply static tf
    in_frame = transformations.euler_affine_rotate(in_frame, np.array([args.yaw, args.pitch, args.roll]), args.degrees)

    # Set mosaic to first frame
    mosaic = in_frame

    # Window names for output monitoring
    windowNameLive = "Video Input"
    windowNameMosaic = "Mosaic Output"

    # create windows by name (as resizable)
    cv2.namedWindow(windowNameLive, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameMosaic, cv2.WINDOW_NORMAL)
    tfs = []
    while (keep_processing):
        if (cap.isOpened):
            ## Read all the other frames
            ret, in_frame = cap.read()
            # when we reach the end of the video (file) exit cleanly
            if not ret:
                keep_processing = False
                continue

            showImageKeepRatio(windowNameLive, in_frame, 0.5)

            ## Frame Processsing
            # for the video
            input_frame = preprocessing.const_ar_scale(in_frame, 0.6)

            # Fix lighting, colors, contrast here
            in_frame = preprocessing.fix_color(in_frame, 1.0)
            in_frame = preprocessing.fix_light(in_frame)
            in_frame = preprocessing.fix_contrast(in_frame)

            # Apply static tf
            in_frame = transformations.euler_affine_rotate(in_frame, np.array([args.yaw, args.pitch, args.roll]),
                                                           args.degrees)

            ## Image Registration and Stitching
            try:
                success, result, tf = registration.alignImages(mosaic, in_frame,  detector)
            except Exception as e:
                print("Something went wrong in align Images {}".format(e))
                continue

            if success:
                mosaic = result
            else:
                tfs.append(tf)
                print("Dumping Mosaic, Saving TF")
                io.imsave(outputpath.parent.joinpath("tmp_{0:03d}.png".format(len(tfs))), mosaic)
                mosaic = in_frame

            showImageKeepRatio(windowNameMosaic, mosaic, 0.7)

        # continue to next frame (i.e. next loop iteration)

        key = cv2.waitKey(10) & 0xFF
        if (key == ord('x')):
            keep_processing = False
            cv2.destroyWindow(windowNameMosaic)
            cv2.destroyWindow(windowNameLive)

    # writer.release()
    ## Save mosaic
    cv2.imwrite('Final Mosaic.jpg', mosaic)

    end = time.time()
    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))
    cap.release()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Perform mosaicking on video.")
    parser.add_argument("directory",
                        help="Path to video.")
    parser.add_argument("-p", "--pitch", nargs='?', default=0., type=float, help="Static pitch angle to use.")
    parser.add_argument("-r", "--roll", nargs='?', default=0., type=float, help="Static roll angle to use.")
    parser.add_argument("-y", "--yaw", nargs='?', default=0., type=float, help="Static yaw angle to use.")
    parser.add_argument("-d", "--degrees", action="store_true")
    parser.add_argument("output", nargs="?", default=False)
    args = parser.parse_args()

    createMosaic(args)
