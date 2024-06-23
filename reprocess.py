import os
import cv2
import csv
import argparse
import numpy as np
from tqdm import tqdm

def read_rally_scenes(csv_file):
    rally_scenes = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame = int(row['frame'])
            is_rally_scene = row['is_rally_scene'].lower() == 'true'
            rally_scenes[frame] = is_rally_scene
    return rally_scenes

def read_shuttle_positions(csv_file):
    shuttle_positions = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame = int(row['Frame'])
            visibility = int(row['Visibility'])
            x = int(row['X'])
            y = int(row['Y'])
            shuttle_positions[frame] = {'visibility': visibility, 'x': x, 'y': y}
    return shuttle_positions

def read_court_coordinates(file_path):
    with open(file_path, 'r') as file:
        return [tuple(map(float, line.strip().split(';'))) for line in file.readlines()[:4]]

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

def combine_data(rally_scenes, shuttle_positions, total_frames, fps):
    combined_data = []
    for frame in range(total_frames):
        rally_scene = rally_scenes.get(frame, False)
        shuttle_data = shuttle_positions.get(frame, {'visibility': 0, 'x': 0, 'y': 0})
        combined_data.append({
            'frame': frame,
            'is_rally_scene': rally_scene,
            'visibility': shuttle_data['visibility'],
            'x': shuttle_data['x'],
            'y': shuttle_data['y']
        })

    smoothed_results = smooth_decisions([data['is_rally_scene'] for data in combined_data], window_size=5, fps=fps)
    for i, data in enumerate(combined_data):
        data['is_rally_scene'] = smoothed_results[i]

    return combined_data

def save_combined_data(output_file, combined_data):
    with open(output_file, 'w', newline='') as file:
        fieldnames = ['frame', 'is_rally_scene', 'visibility', 'x', 'y']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in combined_data:
            writer.writerow(data)

def annotate_video(input_video, combined_data, output_video, court_corners):
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_data in tqdm(combined_data, desc="Annotating video"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_data['is_rally_scene']:
            for i in range(len(court_corners)):
                cv2.line(frame, tuple(map(int, court_corners[i])), tuple(map(int, court_corners[(i + 1) % len(court_corners)])), (0, 255, 0), 2)

        if frame_data['is_rally_scene'] and frame_data['visibility']:
            cv2.circle(frame, (frame_data['x'], frame_data['y']), 5, (0, 0, 255), -1)

        out.write(frame)

    cap.release()
    out.release()

def process_video(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = "result/"
    os.makedirs(result_dir, exist_ok=True)

    rally_csv = os.path.join(result_dir, f"{base_name}_rally_scenes.csv")
    shuttle_csv = os.path.join(result_dir, f"{base_name}_ball.csv")
    court_file = os.path.join(result_dir, f"{base_name}_court.txt")

    output_video = os.path.join(result_dir, f"{base_name}_processed.mp4")
    output_csv = os.path.join(result_dir, f"{base_name}_combined.csv")

    if not (os.path.exists(rally_csv) and os.path.exists(shuttle_csv) and os.path.exists(court_file)):
        print("Error: Required input files not found.")
        return

    rally_scenes = read_rally_scenes(rally_csv)
    shuttle_positions = read_shuttle_positions(shuttle_csv)
    court_corners = read_court_coordinates(court_file)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    combined_data = combine_data(rally_scenes, shuttle_positions, total_frames, fps)
    save_combined_data(output_csv, combined_data)
    annotate_video(video_path, combined_data, output_video, court_corners)
    print(f"Processed video saved to {output_video}")
    print(f"Combined data saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Process video to annotate rally scenes and shuttle positions.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    args = parser.parse_args()

    process_video(args.video_path)

if __name__ == "__main__":
    main()
