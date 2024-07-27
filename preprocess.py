import cv2
import numpy as np
import csv
import os
import subprocess
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def read_court_coordinates(file_path):
    print(f"Reading court coordinates from {file_path}")
    with open(file_path, 'r') as file:
        coordinates = [tuple(map(float, line.strip().split(';'))) for line in file.readlines()[:4]]
    print(f"Court coordinates: {coordinates}")
    return coordinates

def get_area_hsv_values(hsv_frame, court_corners, area_width=5):
    print("Extracting HSV values around court lines")
    hsv_values = []
    for i in tqdm(range(len(court_corners)), desc="Processing court corners"):
        p1, p2 = court_corners[i], court_corners[(i + 1) % len(court_corners)]
        num_samples = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        x_values = np.linspace(p1[0], p2[0], num_samples).astype(int)
        y_values = np.linspace(p1[1], p2[1], num_samples).astype(int)
        for x, y in zip(x_values, y_values):
            sample_area = hsv_frame[max(0, y-area_width):min(hsv_frame.shape[0], y+area_width+1),
                          max(0, x-area_width):min(hsv_frame.shape[1], x+area_width+1)]
            hsv_values.extend(sample_area.reshape(-1, 3))
    print(f"Extracted {len(hsv_values)} HSV values")
    return np.array(hsv_values)

def determine_hsv_thresholds(hsv_values):
    print("Calculating HSV thresholds")
    mean_hsv = np.mean(hsv_values, axis=0)
    std_hsv = np.std(hsv_values, axis=0)
    lower_bound = np.clip(mean_hsv - 2.5 * std_hsv - 15, 0, 255)
    upper_bound = np.clip(mean_hsv + 2.5 * std_hsv + 15, 0, 255)
    print(f"Lower HSV threshold: {lower_bound}")
    print(f"Upper HSV threshold: {upper_bound}")
    return lower_bound.astype(int), upper_bound.astype(int)

def check_court_presence(hsv_frame, court_corners, lower_hsv, upper_hsv, area_width=5):
    total_in_range = 0
    total_samples = 0
    for i in range(len(court_corners)):
        p1, p2 = court_corners[i], court_corners[(i + 1) % len(court_corners)]
        num_samples = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        x_values = np.linspace(p1[0], p2[0], num_samples).astype(int)
        y_values = np.linspace(p1[1], p2[1], num_samples).astype(int)

        for x, y in zip(x_values, y_values):
            sample_area = hsv_frame[max(0, y-area_width):min(hsv_frame.shape[0], y+area_width+1),
                          max(0, x-area_width):min(hsv_frame.shape[1], x+area_width+1)]
            mask = np.all((lower_hsv <= sample_area) & (sample_area <= upper_hsv), axis=2)
            total_in_range += np.sum(mask)
            total_samples += mask.size

    return total_in_range / total_samples > 0.4

def smooth_decisions(decisions, window_size, fps):
    print("Smoothing and filtering decisions")
    decisions = np.array(decisions, dtype=np.float32)
    kernel = np.ones(window_size) / window_size
    smoothed_decisions = np.convolve(decisions, kernel, mode='same')
    smoothed_decisions = smoothed_decisions > 0.5

    min_frames = int(3 * fps)
    filtered_decisions = np.copy(smoothed_decisions)

    print("Performing forward pass")
    current_duration = 0
    for i in tqdm(range(len(smoothed_decisions)), desc="Forward pass"):
        if smoothed_decisions[i]:
            current_duration += 1
        else:
            if 0 < current_duration < min_frames:
                filtered_decisions[i - current_duration:i] = False
            current_duration = 0

    print("Performing backward pass")
    current_duration = 0
    for i in tqdm(range(len(smoothed_decisions) - 1, -1, -1), desc="Backward pass"):
        if smoothed_decisions[i]:
            current_duration += 1
        else:
            if 0 < current_duration < min_frames:
                filtered_decisions[i + 1:i + current_duration + 1] = False
            current_duration = 0

    print(f"Filtered {np.sum(smoothed_decisions != filtered_decisions)} frames")
    return filtered_decisions.tolist()

def run_detect_script(video_path, output_path):
    print("Running detect script")
    max_retries = 5
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}/{max_retries}")
        result = subprocess.run(f'./resources/detect {video_path} {output_path}', shell=True, capture_output=True, text=True)
        print("Detect script output:")
        print(result.stdout)
        print("Detect script errors:")
        print(result.stderr)
        if "Processing error: Not enough line candidates were found." not in result.stdout:
            break
        print(f"Processing error detected. Retrying...")
    else:
        raise RuntimeError("Failed to process the video after multiple attempts")

    for line in result.stdout.splitlines():
        if "Reading frame with index" in line:
            frame_index = int(line.split()[-1])
            print(f"Detected court in frame {frame_index}")
            return frame_index
    raise ValueError("Failed to extract frame index from detect script output")

def process_frame(frame_index, frame, court_corners, lower_hsv, upper_hsv, area_width=5):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return frame_index, check_court_presence(hsv_frame, court_corners, lower_hsv, upper_hsv, area_width)

def process_video(video_path):
    print(f"Processing video: {video_path}")
    base_name, result_dir = os.path.splitext(os.path.basename(video_path))[0], "result/"
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved in {result_dir}")
    coordinates_file = os.path.join(result_dir, f"{base_name}_court.txt")

    frame_index = run_detect_script(video_path, coordinates_file)

    print("Opening video file")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Error: Could not read frame at index {frame_index}")

    court_corners = read_court_coordinates(coordinates_file)
    print("Converting frame to HSV")
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_values = get_area_hsv_values(hsv_frame, court_corners)
    lower_hsv, upper_hsv = determine_hsv_thresholds(hsv_values)

    print("Reopening video for full processing")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video")

    fourcc, fps = cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {width}x{height} @ {fps} FPS")
    out = cv2.VideoWriter(os.path.join(result_dir, f"{base_name}_processed.mp4"), fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames}")

    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} workers for parallel processing")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for frame_index in tqdm(range(total_frames), desc="Submitting frames for processing"):
            ret, frame = cap.read()
            if not ret:
                print(f"Reached end of video at frame {frame_index}")
                break
            futures.append(executor.submit(process_frame, frame_index, frame, court_corners, lower_hsv, upper_hsv))

        frame_results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
            frame_results.append(future.result())

    print("Sorting frame results")
    frame_results.sort(key=lambda x: x[0])

    print("Smoothing results")
    smoothed_results = smooth_decisions([result[1] for result in frame_results], window_size=5, fps=fps)

    print("Writing output video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_index in tqdm(range(total_frames), desc="Writing output video"):
        ret, frame = cap.read()
        if not ret:
            print(f"Reached end of video at frame {frame_index} while writing")
            break
        if smoothed_results[frame_index]:
            for i in range(len(court_corners)):
                cv2.line(frame, tuple(map(int, court_corners[i])), tuple(map(int, court_corners[(i + 1) % len(court_corners)])), (0, 255, 0), 2)
        out.write(frame)
    cap.release()
    out.release()

    print("Writing results to CSV")
    csv_path = os.path.join(result_dir, f"{base_name}_preprocessed.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "is_rally_scene"])
        for frame, result in tqdm(enumerate(smoothed_results), total=len(smoothed_results), desc="Writing CSV"):
            writer.writerow([frame, result])
    print(f"CSV file saved: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video to detect rally scenes.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    args = parser.parse_args()
    process_video(args.video_path)
    print(f"Rally scenes results saved to result/")