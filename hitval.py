import cv2
import os
import sys
import argparse
import numpy as np

def load_hit_frames(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            return [int(float(x)) for x in content.split(',')]
    except ValueError as e:
        print(f"Error reading hit frames from {file_path}: {e}")
        return []
    except IOError as e:
        print(f"Error opening file {file_path}: {e}")
        return []


def compare_hit_files(hitval_file, hit_file, tolerance):
    hitval_frames = load_hit_frames(hitval_file)
    hit_frames = load_hit_frames(hit_file)

    hitval_set = set(hitval_frames)
    hit_set = set(hit_frames)

    matched = []
    unmatched_hitval = []
    unmatched_hit = []

    for frame in hitval_frames:
        if any(abs(frame - h) <= tolerance for h in hit_set):
            matched.append(frame)
        else:
            unmatched_hitval.append(frame)

    for frame in hit_frames:
        if not any(abs(frame - h) <= tolerance for h in hitval_set):
            unmatched_hit.append(frame)

    return matched, unmatched_hitval, unmatched_hit

def print_comparison_results(matched, unmatched_hitval, unmatched_hit):
    print("\nComparison Results:")
    print("===================")
    print(f"Matched Frames: {len(matched)}")
    print(f"Unmatched Frames in hitval: {len(unmatched_hitval)}")
    print(f"Unmatched Frames in hit: {len(unmatched_hit)}")

    print("\nDetailed Results:")
    print("----------------")
    print("Matched Frames:", ', '.join(map(str, matched)))
    print("Unmatched Frames in hitval:", ', '.join(map(str, unmatched_hitval)))
    print("Unmatched Frames in hit:", ', '.join(map(str, unmatched_hit)))

def annotate_hits(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs('result', exist_ok=True)

    hitval_file = f"result/{base_name}_hitval.txt"
    hit_file = f"result/{base_name}_hit.txt"

    if os.path.exists(hitval_file):
        print(f"Hitval file already exists: {hitval_file}")
        if os.path.exists(hit_file):
            print(f"Comparing with hit file: {hit_file}")
            matched, unmatched_hitval, unmatched_hit = compare_hit_files(hitval_file, hit_file, tolerance=15)  # Changed tolerance to 15
            print_comparison_results(matched, unmatched_hitval, unmatched_hit)
        else:
            print(f"Hit file not found: {hit_file}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    hit_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('h'):
            hit_frames.append(frame_count)
            print(f"Hit recorded at frame {frame_count}")
        elif key == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    with open(hitval_file, 'w') as f:
        f.write(','.join(map(str, hit_frames)))

    print(f"Hit frames saved to {hitval_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate hit frames in a video")
    parser.add_argument("video_path", help="Path to the input video file")
    args = parser.parse_args()

    annotate_hits(args.video_path)
