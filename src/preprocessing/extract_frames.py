import argparse
import os
from pathlib import Path

import cv2


def extract_frames(video_path, frames_dir):
    video_name = Path(video_path).stem

    video_frames_dir = os.path.join(frames_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(video_frames_dir, f"{frame_count:06d}.jpg")

        cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {video_frames_dir}")
    return video_name, frame_count


def main(videos_dir, frames_dir, annotations_file):
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(annotations_file), exist_ok=True)

    with open(annotations_file, "w") as f:
        # Loop through all the video files in the video directory
        for video_file in os.listdir(videos_dir):
            if video_file.endswith(".avi") or video_file.endswith(".mp4"):
                video_path = os.path.join(videos_dir, video_file)
                video_name, num_frames = extract_frames(video_path, frames_dir)
                f.write(f"{video_name} 0 {num_frames - 1} 0\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="Directory path to the videos.",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory path to the frames.",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        required=True,
        help="Path to the annotations file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.videos_dir, args.frames_dir, args.annotations_file)
