import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
import sys
import numpy as np
from mosaicking.preprocessing import fix_color, fix_contrast, fix_light
from skimage import transform

"""In this method, the new image is registered to a tile aggregator.
When the aggregator reaches a critical size (max_tile_size parameter), or the image cannot be registered,
the aggregator is stored along with a mask of the location of the last image registered
a mask of the first image used in the tile. When the video is finished, the stored tiles are registered
in sequence matching the features in the corresponding masks from the last and first."""


def get_features(img: np.ndarray, fdet: cv2.Feature2D, mask=None):
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return fdet.detectAndCompute(img, mask)


def get_starting_pos(cap: cv2.VideoCapture, args):
    if args.start_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000.0)
    elif args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame - 1)
    return cap


def evaluate_stopping(cap: cv2.VideoCapture, args):
    """Return true if stopping conditions met."""
    if args.finish_time:
        return cap.get(cv2.CAP_PROP_POS_MSEC) > args.finish_time*1000.0
    elif args.finish_frame:
        return cap.get(cv2.CAP_PROP_POS_FRAMES) > args.finish_frame-1
    return cap.get(cv2.CAP_PROP_FRAME_COUNT)-1 <= cap.get(cv2.CAP_PROP_POS_FRAMES)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to video file.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start_time", type=float, help="Time (secs) to start from.")
    group.add_argument("--start_frame", type=int, help="Frame number to start from.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--finish_time", type=float, help="Time (secs) to finish at.")
    group.add_argument("--finish_frame", type=int, help="Frame number to finish at.")
    parser.add_argument("--frame_skip", type=int, default=None, help="Number of frames to skip between each mosaic update.")
    parser.add_argument("--tile_size", type=int, default=None, help="Maximum size of a tile before subdividing.")
    parser.add_argument("--min_count", type=int, default=10, help="Minimum number of good matches before subdividing.")
    parser.add_argument("--visualize", action="store_true", help="Flag to visualize the mosaic (high memory usage).")
    parser.add_argument("--map_output", action="store_true", help="Flag to save mosaic generation as video.")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    assert video_path.exists(), "File not found: {}".format(str(video_path))

    reader = cv2.VideoCapture(str(video_path))
    reader = get_starting_pos(reader, args)
    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    formatspec = "{:0"+"{}d".format(len(str(n_frames)))+"}"

    if args.visualize:
        cv2.namedWindow(str(video_path), cv2.WINDOW_AUTOSIZE)
        if args.map_output:
            cv2.namedWindow("OUTPUT", cv2.WINDOW_AUTOSIZE)

    if args.map_output:
        writer = cv2.VideoWriter(str(video_path.parent.joinpath("MOSAIC_"+video_path.name)),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 reader.get(cv2.CAP_PROP_FPS),
                                 (height, width*2))

    # Use the Oriented - BRIEF detector because it is open source :)
    orb = cv2.ORB_create(nfeatures=500)
    # BUT! SIFT is much better!
    orb = cv2.SIFT_create(nfeatures=500)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # How many good matches are required
    MIN_MATCH_COUNT = args.min_count
    MAX_TILE_SIZE = args.tile_size if args.tile_size is not None else width*height*4

    tiles = []
    first = True

    while not evaluate_stopping(reader, args):
        # Skip ahead frames if frame_skip argument given
        if args.frame_skip is not None and not first:
            reader.set(cv2.CAP_PROP_POS_FRAMES, reader.get(cv2.CAP_PROP_POS_FRAMES)+args.frame_skip-1)
        # Acquire the frame
        ret, img = reader.read()
        # Skip the frame if the frame is corrupt/missing.
        if not ret:
            sys.stderr.write("Frame missing: {}\n".format(formatspec.format(int(reader.get(cv2.CAP_PROP_POS_FRAMES)))))
            continue
        elif first:
            # Detect keypoints and compute descriptors of the tile
            tile_img = img.copy()
            tile_mask = np.ones((height, width), np.uint8)
            last_mask = np.ones((height, width), np.uint8)
            first_mask = np.ones((height, width), np.uint8)
            first = False
            continue

        sys.stdout.write("Processing frame: {}\n".format(formatspec.format(int(reader.get(cv2.CAP_PROP_POS_FRAMES)))))
        # Preprocess the image
        # img = fix_color(img)
        # img = fix_light(img)
        img = fix_contrast(img)

        # Detect keypoints
        kp_tile, des_tile = get_features(tile_img, orb, mask=last_mask)
        kp, des = get_features(img, orb)

        if des.shape[0] < MIN_MATCH_COUNT:
            sys.stderr.write("Not enough features.\n")
            continue

        # Match keypoints
        knn_matches = flann.knnMatch(des_tile.astype(np.float32), des.astype(np.float32), 2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in knn_matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if not len(good) > MIN_MATCH_COUNT or np.prod(tile_img.shape[:2]) > MAX_TILE_SIZE:
            if not len(good) > MIN_MATCH_COUNT:
                sys.stderr.write("Not Enough Matches.\n")
            elif np.prod(tile_img.shape[:2]) > MAX_TILE_SIZE:
                sys.stderr.write("Maximum tile size reached.\n")
            tiles.append({"tile": tile_img.copy(),
                          "first": first_mask.copy(),
                          "last": last_mask.copy()})
            tile_img = img.copy()
            tile_mask = np.ones((height, width), np.uint8)
            last_mask = np.ones((height, width), np.uint8)
            first_mask = np.ones((height, width), np.uint8)
            continue

        # Source Image
        src_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # Destination Image is the previous
        dst_pts = np.float32([ kp_tile[m.queryIdx].pt for m in good ]).reshape(-1,1,2)

        # Get the homography from current image to previous
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # draw_matches(prev_img, kp_prev, img, kp, good, matchesMask)

        # Get the corners of the current image in homogeneous coords (X,Y,Z=0,W=1)
        src_crn = np.array([[0, width, width, 0],
                            [0, 0, height, height],
                            [1,1,1,1]], np.float)
        # Get the corners of the mosaic image in homogeneous coords (X,Y,Z=0,W=1)
        dst_crn = np.array([[0, tile_img.shape[1], tile_img.shape[1], 0],
                            [0, 0, tile_img.shape[0], tile_img.shape[0]],
                            [1, 1, 1, 1]], np.float)

        # Warp the src corners to get them into the same plane as dst
        warp_dst = H @ src_crn

        # Concatenate
        pts = np.concatenate([dst_crn, warp_dst], axis=1)
        # Convert to cartesian coords
        pts = pts[:2, :]/pts[-1, :]

        # Find minimum and maximum bounds of the composite tile
        xmin,ymin = np.int32(pts.min(axis=1) - 0.5)
        xmax,ymax = np.int32(pts.max(axis=1) + 0.5)

        # The translation operator for tile
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translation homography

        # Get the mask of the last aggregated image
        last_mask = cv2.warpPerspective(np.ones((height, width), np.uint8), Ht.dot(H), (xmax - xmin, ymax - ymin))

        # Get the previous iteration tile_mask in the shape of the update
        template = cv2.warpPerspective(np.zeros(img.shape, np.uint8), Ht.dot(H), (xmax - xmin, ymax - ymin))
        tile_mask_ = template[:,:,0].copy()
        tile_mask_[t[1]:tile_mask.shape[0] + t[1], t[0]:tile_mask.shape[1] + t[0]] = tile_mask
        tile_img_ = template.copy()
        tile_img_[t[1]:tile_img.shape[0]+t[1], t[0]:tile_img.shape[1]+t[0]] = tile_img
        first_mask_ = template[:,:,0].copy()
        first_mask_[t[1]:first_mask.shape[0]+t[1], t[0]:first_mask.shape[1]+t[0]] = first_mask
        first_mask = first_mask_.copy()

        # warp the input image into the tile's plane
        warped = cv2.warpPerspective(img, Ht.dot(H), (xmax - xmin, ymax - ymin))

        # Get img only
        img_only = cv2.bitwise_and(last_mask, cv2.bitwise_not(tile_mask_))
        # Tile only
        tile_only = cv2.bitwise_and(tile_mask_, cv2.bitwise_not(last_mask))
        # intersection
        shared = cv2.bitwise_and(last_mask, tile_mask_)

        # Combine the image and tile, and update
        tile_img = np.where(cv2.cvtColor(img_only, cv2.COLOR_GRAY2BGR) > 0, warped, tile_img_)
        tile_img = np.where(cv2.cvtColor(tile_only, cv2.COLOR_GRAY2BGR) > 0, tile_img_, tile_img)
        mixer = np.uint8(0.5*tile_img_.astype(np.float32)+0.5*warped.astype(np.float32))
        tile_img = np.where(cv2.cvtColor(shared, cv2.COLOR_GRAY2BGR) > 0, mixer, tile_img)

        # update the tile_mask
        tile_mask = cv2.bitwise_or(tile_mask_, last_mask)

        if args.map_output:
            map_out = np.concatenate([img.copy(), cv2.resize(tile_img, tuple(reversed(img.shape[:2])))], axis=1)
            writer.write(map_out)
            if args.visualize:
                cv2.imshow("OUTPUT", map_out)

        # show tile
        # cv2.imshow(str(video_path), (first_mask>0).astype(np.uint8)*255)
        # cv2.imshow(str(video_path), (last_mask > 0).astype(np.uint8) * 255)
        # After image processing is done
        # img = cv2.putText(img, "{:.2f}".format(reader.get(cv2.CAP_PROP_POS_MSEC)/1000.0),
        #                  (0, height-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        # img = cv2.putText(img, "{}".format(formatspec.format(int(reader.get(cv2.CAP_PROP_POS_FRAMES)+1))),
        #                  (0, height-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        # cv2.imshow(str(video_path), img)
        if args.visualize:
            cv2.imshow(str(video_path), tile_img)
            key = cv2.waitKey(10)
            if key == ord("q"):
                sys.stdout.write("Quitting.\n")
                break
    tiles.append({"tile": tile_img.copy(),
                  "first": first_mask.copy(),
                  "last": last_mask.copy()})
    if args.visualize:
        cv2.destroyAllWindows()
    if args.map_output:
        writer.release()
    reader.release()
    output_path = Path("./tmp")
    output_path.mkdir(parents=True, exist_ok=True)
    formatspec = "{:"+"0{}d".format(len(str(len(tiles))))+"}"
    for i, tile in enumerate(tiles):
        fpath = output_path.joinpath("tile_{}.png".format(formatspec.format(i)))
        cv2.imwrite(str(fpath), tile["tile"])
        fpath = output_path.joinpath("first_{}.png".format(formatspec.format(i)))
        cv2.imwrite(str(fpath), tile["first"])
        fpath = output_path.joinpath("last_{}.png".format(formatspec.format(i)))
        cv2.imwrite(str(fpath), tile["last"])