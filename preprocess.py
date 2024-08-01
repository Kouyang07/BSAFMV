import logging
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
    coordinates = {}
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            coordinates[row['Point']] = (float(row['X']), float(row['Y']))
    print(f"Court coordinates: {coordinates}")
    return coordinates

def get_area_hsv_values(hsv_frame, court_points, area_width=5):
    print("Extracting HSV values around court lines")
    hsv_values = []
    court_lines = [
        ('P1', 'P2'), ('P2', 'P3'), ('P3', 'P4'), ('P4', 'P1'),
        ('P5', 'P6'), ('P7', 'P8'), ('P9', 'P10'), ('P11', 'P12'),
        ('P13', 'P14'), ('P15', 'P16'), ('P17', 'P18'), ('P19', 'P20'),
        ('P21', 'P22')
    ]
    for start, end in tqdm(court_lines, desc="Processing court lines"):
        p1, p2 = court_points[start], court_points[end]
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

def check_court_presence(hsv_frame, court_points, lower_hsv, upper_hsv, area_width=5):
    total_in_range = 0
    total_samples = 0
    court_lines = [
        ('P1', 'P2'), ('P2', 'P3'), ('P3', 'P4'), ('P4', 'P1'),
        ('P5', 'P6'), ('P7', 'P8'), ('P9', 'P10'), ('P11', 'P12'),
        ('P13', 'P14'), ('P15', 'P16'), ('P17', 'P18'), ('P19', 'P20'),
        ('P21', 'P22')
    ]
    for start, end in court_lines:
        p1, p2 = court_points[start], court_points[end]
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
    max_retries = 15
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

def process_frame(frame_index, frame, court_points, lower_hsv, upper_hsv, area_width=5):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return frame_index, check_court_presence(hsv_frame, court_points, lower_hsv, upper_hsv, area_width)

def process_video(video_path):
    logging.info(f"Processing video: {video_path}")
    base_name, result_dir = os.path.splitext(os.path.basename(video_path))[0], "result/"
    os.makedirs(result_dir, exist_ok=True)
    logging.info(f"Results will be saved in {result_dir}")
    coordinates_file = os.path.join(result_dir, f"{base_name}_court.csv")

    frame_index = run_detect_script(video_path, coordinates_file)

    logging.info("Opening video file")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Error: Could not read frame at index {frame_index}")

    court_points = read_court_coordinates(coordinates_file)
    logging.info("Converting frame to HSV")
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_values = get_area_hsv_values(hsv_frame, court_points)
    lower_hsv, upper_hsv = determine_hsv_thresholds(hsv_values)

    logging.info("Reopening video for full processing")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video")

    fourcc, fps = cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"Video properties: {width}x{height} @ {fps} FPS")
    out = cv2.VideoWriter(os.path.join(result_dir, f"{base_name}_processed.mp4"), fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Total frames to process: {total_frames}")

    num_workers = multiprocessing.cpu_count()
    logging.info(f"Using {num_workers} workers for parallel processing")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for frame_index in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                logging.info(f"Reached end of video at frame {frame_index}")
                break
            futures.append(executor.submit(process_frame, frame_index, frame, court_points, lower_hsv, upper_hsv))

        frame_results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing frames for post-processing"):
            frame_results.append(future.result())

    logging.info("Sorting frame results")
    frame_results.sort(key=lambda x: x[0])

    logging.info("Smoothing results")
    smoothed_results = smooth_decisions([result[1] for result in frame_results], window_size=5, fps=fps)

    logging.info("Writing output video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    court_lines = [
        ('P1', 'P2'), ('P2', 'P3'), ('P3', 'P4'), ('P4', 'P1'),
        ('P5', 'P6'), ('P7', 'P8'), ('P9', 'P10'), ('P11', 'P12'),
        ('P13', 'P21'), ('P14', 'P22'), ('P17', 'P18'), ('P19', 'P20')
        # Removed ('P21', 'P22')
    ]
    for frame_index in tqdm(range(total_frames), desc="Writing output video"):
        ret, frame = cap.read()
        if not ret:
            logging.info(f"Reached end of video at frame {frame_index} while writing")
            break
        if smoothed_results[frame_index]:
            for start, end in court_lines:
                cv2.line(frame, tuple(map(int, court_points[start])), tuple(map(int, court_points[end])), (0, 255, 0), 2)
        out.write(frame)
    cap.release()
    out.release()

    logging.info("Writing results to CSV")
    csv_path = os.path.join(result_dir, f"{base_name}_preprocessed.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "is_rally_scene"])
        for frame, result in tqdm(enumerate(smoothed_results), total=len(smoothed_results), desc="Writing CSV"):
            writer.writerow([frame, result])
    logging.info(f"CSV file saved: {csv_path}")

def main(video_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    process_video(video_path)
    logging.info(f"Rally scenes results saved to result/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video to detect rally scenes.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    args = parser.parse_args()
    main(args.video_path)