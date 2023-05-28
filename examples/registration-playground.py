from mosaicking import registration, transformations
import argparse
import cv2
from pathlib import Path
import numpy as np


def add_inner_border(image, thickness, color):
    # Draw top border
    cv2.rectangle(image, (0, 0), (image.shape[1], thickness), color, thickness)
    # Draw bottom border
    cv2.rectangle(image, (0, image.shape[0]), (image.shape[1], image.shape[0] - thickness), color, thickness)
    # Draw left border
    cv2.rectangle(image, (0, 0), (thickness, image.shape[0]), color, thickness)
    # Draw right border
    cv2.rectangle(image, (image.shape[1], 0), (image.shape[1] - thickness, image.shape[0]), color, thickness)
    return image


def main():
    parser = argparse.ArgumentParser(description='Demonstrate the effects of image preprocessing functions.')
    parser.add_argument('input_images', type=Path, nargs="+", help='Paths to input images.')
    parser.add_argument('--n_features', type=int, default=None, help="Maximum number of features to detect in images.")
    args = parser.parse_args()

    assert len(args.input_images) > 1, f"Need to provide more than 1 image, received {len(args.input_images)}"
    [path.resolve(True) for path in args.input_images]

    # Load the input images
    imgs = np.stack(tuple(cv2.imread(str(path)) for path in args.input_images), axis=0)

    # Load the feature detector
    detector = cv2.SIFT_create(args.n_features)

    # Load the feature matcher
    matcher = cv2.FlannBasedMatcher_create()

    # Detect features in the images
    feature_set = tuple(registration.get_features(img, detector) for img in imgs)

    img_pairs = []
    match_imgs = []
    registered_imgs = []
    cv2.namedWindow("imgs", cv2.WINDOW_NORMAL)
    cv2.namedWindow("registered", cv2.WINDOW_NORMAL)
    cv2.namedWindow("matches", cv2.WINDOW_NORMAL)
    cv2.setWindowTitle("imgs", "Image Pair")
    cv2.setWindowTitle("registered", "Registration Result")
    cv2.setWindowTitle("matches", "Matched Features")
    # Match features from each image
    for src_idx, (keypoints_src, descriptors_src) in enumerate(feature_set[:-1]):
        for i, (keypoints_dst, descriptors_dst) in enumerate(feature_set[src_idx + 1:]):
            is_valid, matches = registration.get_matches(descriptors_src, descriptors_dst, matcher, 50)
            tmp = np.zeros_like(imgs[src_idx])
            src_pts, dst_pts = registration.get_match_points(keypoints_src, keypoints_dst, matches)
            M, xbounds, ybounds = transformations.get_alignment(src_pts, imgs[src_idx].shape[:2], dst_pts, imgs[i + src_idx + 1].shape[:2], 'perspective')
            imgA = add_inner_border(imgs[src_idx], 5, (0, 255, 0))
            imgB = add_inner_border(imgs[i + src_idx + 1], 5, (0, 0, 255))
            img_pairs.append(np.concatenate((imgA, imgB), axis=1))
            match_imgs.append(cv2.drawMatches(imgA, keypoints_src, imgB, keypoints_dst, matches, tmp))
            warped_src = cv2.warpPerspective(imgA, M, (xbounds[1]-xbounds[0], ybounds[1]-ybounds[0]))
            t = [-min(xbounds), -min(ybounds)]
            warped_dst = np.zeros_like(warped_src)
            warped_dst[t[1]:imgB.shape[0] + t[1], t[0]:imgB.shape[1] + t[0]] = imgB
            registered_imgs.append(cv2.addWeighted(warped_src, 0.5, warped_dst, 0.5, 1.0))
        cv2.imshow("imgs", img_pairs[src_idx])
        cv2.imshow("matches", match_imgs[src_idx])
        cv2.imshow("registered", registered_imgs[src_idx])
        key = -1
        while key & 0xFF != 27 and bool(cv2.getWindowProperty("matches", cv2.WND_PROP_VISIBLE)) and bool(cv2.getWindowProperty("registered", cv2.WND_PROP_VISIBLE)) and bool(cv2.getWindowProperty("imgs", cv2.WND_PROP_VISIBLE)):
            key = cv2.waitKey(1)
    cv2.destroyWindow("imgs")
    cv2.destroyWindow("matches")
    cv2.destroyWindow("registered")


if __name__ == "__main__":
    main()
