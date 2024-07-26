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
    with open(file_path, 'r') as file:
        return [tuple(map(float, line.strip().split(';'))) for line in file.readlines()[:4]]

def get_area_hsv_values(hsv_frame, court_corners, area_width=5):
    hsv_values = []
    for i in range(len(court_corners)):
        p1, p2 = court_corners[i], court_corners[(i + 1) % len(court_corners)]
        num_samples = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        x_values = np.linspace(p1[0], p2[0], num_samples).astype(int)
        y_values = np.linspace(p1[1], p2[1], num_samples).astype(int)
        for x, y in zip(x_values, y_values):
            sample_area = hsv_frame[max(0, y-area_width):min(hsv_frame.shape[0], y+area_width+1),
                          max(0, x-area_width):min(hsv_frame.shape[1], x+area_width+1)]
            hsv_values.extend(sample_area.reshape(-1, 3))
    return np.array(hsv_values)

def determine_hsv_thresholds(hsv_values):
    mean_hsv = np.mean(hsv_values, axis=0)
    std_hsv = np.std(hsv_values, axis=0)
    lower_bound = np.clip(mean_hsv - 2.5 * std_hsv - 15, 0, 255)
    upper_bound = np.clip(mean_hsv + 2.5 * std_hsv + 15, 0, 255)
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

def process_frame(frame_index, frame, court_corners, lower_hsv, upper_hsv, area_width=5):
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb_frame = cv2.normalize(ycrcb_frame, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(ycrcb_frame, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength=100, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (x1 >= court_corners[0][0] and x1 <= court_corners[1][0] and
                    y1 >= court_corners[0][1] and y1 <= court_corners[1][1] and
                    x2 >= court_corners[0][0] and x2 <= court_corners[1][0] and
                    y2 >= court_corners[0][1] and y2 <= court_corners[1][1]):
                print(f"Frame {frame_index}: Court detected")
                return frame_index, True
    print(f"Frame {frame_index}: Court not detected")
    return frame_index, False

def smooth_decisions(decisions, window_size, fps):
    decisions = np.array(decisions, dtype=np.float32)
    kernel = np.ones(window_size) / window_size
    smoothed_decisions = np.convolve(decisions, kernel, mode='same')
    smoothed_decisions = smoothed_decisions > 0.5

    min_frames = int(3 * fps)
    filtered_decisions = np.copy(smoothed_decisions)

    current_duration = 0
    for i in tqdm(range(len(smoothed_decisions)), desc="Forward pass"):
        if smoothed_decisions[i]:
            current_duration += 1
        else:
            if 0 < current_duration < min_frames:
                filtered_decisions[i - current_duration:i] = False
            current_duration = 0

    current_duration = 0
    for i in tqdm(range(len(smoothed_decisions) - 1, -1, -1), desc="Backward pass"):
        if smoothed_decisions[i]:
            current_duration += 1
        else:
            if 0 < current_duration < min_frames:
                filtered_decisions[i + 1:i + current_duration + 1] = False
            current_duration = 0

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
        raise RuntimeError(f"Error: Could not read frame at index {frame_index}")

    court_corners = read_court_coordinates(coordinates_file)
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb_frame = cv2.normalize(ycrcb_frame, None, 0, 255, cv2.NORM_MINMAX)
    hsv_values = get_area_hsv_values(ycrcb_frame, court_corners)
    lower_hsv, upper_hsv = determine_hsv_thresholds(hsv_values)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video")

    fourcc, fps = cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(result_dir, f"{base_name}_preprocessed.mp4"), fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    num_workers = multiprocessing.cpu_count()
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

    frame_results.sort(key=lambda x: x[0])

    smoothed_results = smooth_decisions([result[1] for result in frame_results], window_size=5, fps=fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    log_file = open(os.path.join(result_dir, f"{base_name}_log.txt"), 'w')
    for frame_index in tqdm(range(total_frames), desc="Writing output video"):
        ret, frame = cap.read()
        if not ret:
            print(f"Reached end of video at frame {frame_index} while writing")
            break
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb_frame = cv2.normalize(ycrcb_frame, None, 0, 255, cv2.NORM_MINMAX)
        edges = cv2.Canny(ycrcb_frame, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength=100, maxLineGap=10)

        if smoothed_results[frame_index]:
            for i in range(len(court_corners)):
                cv2.line(frame, tuple(map(int, court_corners[i])), tuple(map(int, court_corners[(i + 1) % len(court_corners)])), (0, 255, 0), 2)
            log_file.write(f"Frame {frame_index}: Valid - Court detected\n")
            log_file.write(f"HSV values: {np.mean(hsv_values, axis=0)}\n")
            log_file.write(f"Lower HSV threshold: {lower_hsv}\n")
            log_file.write(f"Upper HSV threshold: {upper_hsv}\n")
            if lines is not None:
                log_file.write(f"Number of lines detected: {len(lines)}\n")
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    log_file.write(f"Line coordinates: ({x1}, {y1}) -> ({x2}, {y2})\n")
            else:
                log_file.write(f"Number of lines detected: 0\n")
            log_file.write(f"Edge detection values: {np.mean(edges)}\n")
            log_file.write(f"Hough transform values: {np.mean(lines) if lines is not None else 0}\n")
            log_file.write(f"HSV values of court: {np.mean(ycrcb_frame)}\n")
            log_file.write(f"RGB values of court: {np.mean(frame)}\n")
        else:
            log_file.write(f"Frame {frame_index}: Invalid - Court not detected\n")
            log_file.write(f"HSV values: {np.mean(hsv_values, axis=0)}\n")
            log_file.write(f"Lower HSV threshold: {lower_hsv}\n")
            log_file.write(f"Upper HSV threshold: {upper_hsv}\n")
            if lines is not None:
                log_file.write(f"Number of lines detected: {len(lines)}\n")
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    log_file.write(f"Line coordinates: ({x1}, {y1}) -> ({x2}, {y2})\n")
            else:
                log_file.write(f"Number of lines detected: 0\n")
            log_file.write(f"Edge detection values: {np.mean(edges)}\n")
            log_file.write(f"Hough transform values: {np.mean(lines) if lines is not None else 0}\n")
            log_file.write(f"HSV values of frame: {np.mean(ycrcb_frame)}\n")
            log_file.write(f"RGB values of frame: {np.mean(frame)}\n")
        out.write(frame)
    cap.release()
    out.release()
    log_file.close()

    csv_path = os.path.join(result_dir, f"{base_name}_preprocessed.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "is_rally_scene"])
        for frame, result in tqdm(enumerate(smoothed_results), total=len(smoothed_results), desc="Writing CSV"):
            writer.writerow([frame, result])
    print(f"CSV file saved: {csv_path}")
    print(f"Output video saved: {result_dir}{base_name}_preprocessed.mp4")
    print(f"Log file saved: {result_dir}{base_name}_log.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video to detect rally scenes.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    args = parser.parse_args()
    process_video(args.video_path)
    print(f"Rally scenes results saved to result/")