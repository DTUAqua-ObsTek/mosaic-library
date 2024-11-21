import cv2
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from mosaicking.preprocessing import ConstARScaling
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mosaic_output", type=Path, help="Path to mosaic tiles.")
    parser.add_argument("--scaling", type=float, default=1.0, help="Scaling factor to apply to tiles.")
    args = parser.parse_args()
    # Define the directory containing the images

    image_dir = args.mosaic_output.resolve(True)
    scaling = args.scaling
    scaler = ConstARScaling(scaling)

    # Function to extract x and y coordinates from the filename
    def extract_info(filename: str) -> tuple[int, int, int]:
        pattern = r"seq_(\d+)_tile_(\d+)_(\d+)\.png"
        match = re.match(pattern, filename)
        if match:
            seq = int(match.group(1))
            x = int(match.group(2))
            y = int(match.group(3))
            return seq, x, y
        return None

    # Get the list of all images
    image_filenames = [f for f in Path(image_dir).glob("seq*.png")]

    tile_h, tile_w = scaler.apply(cv2.imread(str(image_dir / image_filenames[0]))).shape[:2]

    # Group images by sequence number
    images_by_sequence = defaultdict(list)
    for filename in image_filenames:
        info = extract_info(filename.name)
        if info:
            sequence, x, y = info
            images_by_sequence[sequence].append((filename, x, y))

    # For each sequence, create the combined image
    for sequence, image_info in images_by_sequence.items():
        # Extract top left coordinates and find max x and y for canvas size
        max_x = max([x * tile_w for _, x, _ in image_info]) + tile_w
        max_y = max([y * tile_h for _, _, y in image_info]) + tile_h

        # Create a blank canvas (final combined image for this sequence)
        canvas = np.zeros((max_y, max_x, 3), dtype=np.uint8)

        # Load each image and place it at the correct location
        for filename, x, y in image_info:
            img = scaler.apply(cv2.imread(str(filename)))
            # Place the image in the canvas
            canvas[y*tile_h:y*tile_h + tile_h, x*tile_w:x*tile_w + tile_w] = img

        # Save the final combined image for this sequence
        output_path = image_dir / f"combined_image_seq_{sequence}.png"
        cv2.imwrite(str(output_path), canvas)
        print(f"Combined image for sequence {sequence} saved as {output_path}")
