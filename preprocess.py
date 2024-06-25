import cv2
import numpy as np
import csv
import os
import subprocess
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_court_coordinates(file_path):
    with open(file_path, 'r') as file:
        return [tuple(map(float, line.strip().split(';'))) for line in file.readlines()[:4]]

def get_area_hsv_values(frame, court_corners, area_width=5):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_values = []

    def sample_area_hsv(p1, p2, area_width):
        num_samples = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        x_values = np.linspace(p1[0], p2[0], num_samples).astype(int)
        y_values = np.linspace(p1[1], p2[1], num_samples).astype(int)
        for x, y in zip(x_values, y_values):
            for dx in range(-area_width, area_width + 1, area_width // 2):
                for dy in range(-area_width, area_width + 1, area_width // 2):
                    if 0 <= x + dx < hsv_frame.shape[1] and 0 <= y + dy < hsv_frame.shape[0]:
                        hsv_values.append(hsv_frame[y + dy, x + dx])

    for i in range(len(court_corners)):
        sample_area_hsv(court_corners[i], court_corners[(i + 1) % len(court_corners)], area_width)

    return np.array(hsv_values)

def determine_hsv_thresholds(hsv_values):
    mean_hsv = np.mean(hsv_values, axis=0)
    std_hsv = np.std(hsv_values, axis=0)
    lower_bound = mean_hsv - 2 * std_hsv - 10  # Adding extra tolerance
    upper_bound = mean_hsv + 2 * std_hsv + 10  # Adding extra tolerance
    lower_bound[lower_bound < 0] = 0
    upper_bound[upper_bound > 255] = 255
    return lower_bound.astype(int), upper_bound.astype(int)

def check_court_presence(frame, court_corners, lower_hsv, upper_hsv, area_width=5):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def is_area_within_range(p1, p2, area_width):
        num_samples = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        x_values = np.linspace(p1[0], p2[0], num_samples).astype(int)
        y_values = np.linspace(p1[1], p2[1], num_samples).astype(int)
        in_range_count = 0
        total_samples = 0
        for x, y in zip(x_values, y_values):
            for dx in range(-area_width, area_width + 1, area_width // 2):
                for dy in range(-area_width, area_width + 1, area_width // 2):
                    if 0 <= x + dx < hsv_frame.shape[1] and 0 <= y + dy < hsv_frame.shape[0]:
                        total_samples += 1
                        if (lower_hsv <= hsv_frame[y + dy, x + dx]).all() and (hsv_frame[y + dy, x + dx] <= upper_hsv).all():
                            in_range_count += 1
        return in_range_count / total_samples > 0.5  # More lenient condition

    for i in range(len(court_corners)):
        if not is_area_within_range(court_corners[i], court_corners[(i + 1) % len(court_corners)], area_width):
            return False
    return True

def smooth_decisions(decisions, window_size, fps):
    smoothed_decisions = [sum(decisions[max(0, i - window_size // 2):min(len(decisions), i + window_size // 2 + 1)]) / (min(len(decisions), i + window_size // 2 + 1) - max(0, i - window_size // 2)) > 0.5 for i in range(len(decisions))]
    min_frames = int(3 * fps)
    filtered_decisions, current_duration = smoothed_decisions[:], 0
    for i, decision in enumerate(smoothed_decisions):
        if decision: current_duration += 1
        else:
            if 0 < current_duration < min_frames:
                filtered_decisions[i - current_duration:i] = [False] * current_duration
            current_duration = 0
    if 0 < current_duration < min_frames:
        filtered_decisions[-current_duration:] = [False] * current_duration
    return filtered_decisions

def run_detect_script(video_path, output_path):
    print("Running detect script.")
    while "Processing error: Not enough line candidates were found." in (result := subprocess.run(f'./resources/detect {video_path} {output_path}', shell=True, capture_output=True, text=True)).stdout:
        print("Processing error detected. Retrying.")
    for line in result.stdout.splitlines():
        if "Reading frame with index" in line:
            return int(line.split()[-1])
    raise ValueError("Failed to extract frame index from detect script output.")

def process_frame(frame_index, frame, court_corners, lower_hsv, upper_hsv, area_width=5):
    return frame_index, check_court_presence(frame, court_corners, lower_hsv, upper_hsv, area_width)

def process_video(video_path):
    base_name, result_dir = os.path.splitext(os.path.basename(video_path))[0], "result/"
    os.makedirs(result_dir, exist_ok=True)
    coordinates_file = os.path.join(result_dir, f"{base_name}_court.txt")

    frame_index = run_detect_script(video_path, coordinates_file)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame at index {frame_index}.")
        return

    court_corners = read_court_coordinates(coordinates_file)
    hsv_values = get_area_hsv_values(frame, court_corners)
    lower_hsv, upper_hsv = determine_hsv_thresholds(hsv_values)

    cap, frame_results = cv2.VideoCapture(video_path), []

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc, fps, width, height = cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(result_dir, f"{base_name}_processed.mp4"), fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, frame_index, frame, court_corners, lower_hsv, upper_hsv) for frame_index, frame in enumerate((cap.read()[1] for _ in tqdm(range(total_frames), desc="Reading frames")))]
        frame_results = [future.result() for future in tqdm(as_completed(futures), total=total_frames, desc="Processing frames")]

    smoothed_results = smooth_decisions([result[1] for result in sorted(frame_results)], window_size=5, fps=fps)

    cap = cv2.VideoCapture(video_path)
    for frame_index in tqdm(range(total_frames), desc="Writing output video"):
        ret, frame = cap.read()
        if not ret:
            break
        if smoothed_results[frame_index]:
            for i in range(len(court_corners)):
                cv2.line(frame, tuple(map(int, court_corners[i])), tuple(map(int, court_corners[(i + 1) % len(court_corners)])), (0, 255, 0), 2)
        out.write(frame)
    cap.release()
    out.release()

    with open(os.path.join(result_dir, f"{base_name}_processed.csv"), 'w', newline='') as csvfile:
        csv.writer(csvfile).writerow(["frame", "is_rally_scene"])
        csv.writer(csvfile).writerows(enumerate(smoothed_results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video to detect rally scenes.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    args = parser.parse_args()
    process_video(args.video_path)
    print(f"Rally scenes results saved to result/")