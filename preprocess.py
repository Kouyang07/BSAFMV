import cv2
import numpy as np
import csv
import os
import subprocess
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def detect_scoreboard(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 55, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask = np.where(np.isin(labels, [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] < 500]), 0, mask)
    contours, _ = cv2.findContours(cv2.Canny(mask, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            if 2.0 <= w / float(h) <= 5.0:
                return True
    return False

def read_court_coordinates(file_path):
    with open(file_path, 'r') as file:
        return [tuple(map(float, line.strip().split(';'))) for line in file.readlines()[:4]]

def check_court_presence(frame, court_corners, lower_green, upper_green):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, np.array([court_corners], dtype=np.int32), 255)
    roi = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)
    green_ratio = np.sum(np.all((roi[mask == 255] >= lower_green) & (roi[mask == 255] <= upper_green), axis=1)) / roi[mask == 255].shape[0]
    return green_ratio > 0.2

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

def get_court_color(frame):
    return np.array([30, 40, 40]), np.array([90, 255, 255])

def process_frame(frame_index, frame, court_corners, lower_green, upper_green):
    return frame_index, detect_scoreboard(frame) and check_court_presence(frame, court_corners, lower_green, upper_green)

def process_video(video_path):
    base_name, result_dir = os.path.splitext(os.path.basename(video_path))[0], "result/"
    os.makedirs(result_dir, exist_ok=True)
    coordinates_file = os.path.join(result_dir, f"{base_name}_court.txt")

    frame_index = run_detect_script(video_path, coordinates_file)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    lower_green, upper_green = get_court_color(frame) if ret else (None, None)
    cap.release()

    if lower_green is None:
        print(f"Error: Could not read frame at index {frame_index}.")
        return

    court_corners = read_court_coordinates(coordinates_file)
    cap, frame_results = cv2.VideoCapture(video_path), []

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc, fps, width, height = cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(result_dir, f"{base_name}_rally_scenes.mp4"), fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, frame_index, frame, court_corners, lower_green, upper_green) for frame_index, frame in enumerate((cap.read()[1] for _ in tqdm(range(total_frames), desc="Reading frames")))]
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

    with open(os.path.join(result_dir, f"{base_name}_rally_scenes.csv"), 'w', newline='') as csvfile:
        csv.writer(csvfile).writerow(["frame", "is_rally_scene"])
        csv.writer(csvfile).writerows(enumerate(smoothed_results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video to detect rally scenes.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    args = parser.parse_args()
    process_video(args.video_path)
    print(f"Rally scenes results saved to result/")
