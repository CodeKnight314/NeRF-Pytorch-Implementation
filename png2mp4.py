import os
import cv2
import argparse
from glob import glob 
import re

def extract_number(filename):
    """Extract the number from the filename using a regular expression."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def create_video_from_images(image_dir, output_path, frame_rate):
    # Get all image files in the directory and sort them
    images = sorted(glob(os.path.join(image_dir, "*.png")), key=lambda x: extract_number(os.path.basename(x)))
    for img in images: 
        print(img)
    if not images:
        print(f"No images found in directory: {image_dir}")
        return

    first_image_path = images[0]
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    for image_path in images:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error reading image {image_path}, skipping...")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine images into an MP4 video.")
    parser.add_argument("--dir", type=str, help="Directory containing image files.")
    parser.add_argument("--output", type=str, help="Path to save the output MP4 file.")
    parser.add_argument("--frame_rate", type=int, default=15, help="Frame rate for the video (default: 30).")

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"The directory {args.dir} does not exist.")
        exit(1)

    create_video_from_images(args.dir, args.output, args.frame_rate)
