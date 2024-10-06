import os
import cv2
import argparse
import numpy as np
import torch
import csv
from tqdm import tqdm

# Constants
HEIGHT, WIDTH = 288, 512
DETECTION_THRESHOLD = 0.05
SMOOTHING_WINDOW_SIZE = 5  # Window size for moving average smoothing
TELEPORTATION_THRESHOLD = 50  # Threshold distance to consider as teleportation

class Detector:
    def __init__(self, model_file, num_frame, batch_size, device):
        self.model_file = model_file
        self.num_frame = num_frame
        self.batch_size = batch_size
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        from resources.model import TrackNetV2 as TrackNet
        checkpoint = torch.load(self.model_file, map_location=self.device)
        model = TrackNet(in_dim=self.num_frame * 3, out_dim=self.num_frame)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def get_frame_unit(self, frame_list):
        batch = []
        for i in range(0, len(frame_list), self.num_frame):
            frames = frame_list[i:i + self.num_frame]
            if len(frames) < self.num_frame:
                continue
            frames_resized = [cv2.resize(img, (WIDTH, HEIGHT)) for img in frames]
            frames_np = np.array(frames_resized).transpose((0, 3, 1, 2)).reshape(-1, HEIGHT, WIDTH)
            frames_np = frames_np / 255.0  # Normalize to [0,1]
            batch.append(frames_np)

        if len(batch) == 0:
            return None

        batch_np = np.array(batch)
        batch_tensor = torch.FloatTensor(batch_np).to(self.device)
        return batch_tensor

    def get_object_center(self, heatmap):
        contours, _ = cv2.findContours(heatmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return 0, 0
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    def detect(self, frame_buffer):
        x = self.get_frame_unit(frame_buffer)
        if x is None:
            return [(0, 0)] * len(frame_buffer)  # No detection, return empty positions

        with torch.no_grad():
            y_pred = self.model(x)
        h_pred = (y_pred.detach().cpu().numpy() > 0.5).astype('uint8') * 255
        h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

        positions = []
        for i in range(h_pred.shape[0]):
            heatmap = h_pred[i]
            cx, cy = self.get_object_center(heatmap)
            img = frame_buffer[i]
            cx = int(img.shape[1] / WIDTH * cx)
            cy = int(img.shape[0] / HEIGHT * cy)
            positions.append((cx, cy))
        return positions
class PostProcessor:
    def __init__(self, positions):
        self.positions = positions

    def smooth_positions(self):
        # Your existing smoothing code (if any)
        pass

    def detect_teleportations(self):
        corrected_positions = []
        i = 0
        MAX_EXTRAPOLATION_GAP = 5  # Maximum number of frames to extrapolate over
        MIN_POINTS_FOR_EXTRAPOLATION = 3  # Minimum number of points needed before the gap

        while i < len(self.positions):
            curr_pos = self.positions[i]

            if curr_pos != (0, 0):
                corrected_positions.append(curr_pos)
                i += 1
            else:
                # Find the start and end indices of the gap
                gap_start = i
                gap_end = i
                while gap_end < len(self.positions) and self.positions[gap_end] == (0, 0):
                    gap_end += 1

                gap_length = gap_end - gap_start

                if gap_length <= MAX_EXTRAPOLATION_GAP:
                    # Collect positions before the gap
                    before_indices = []
                    b = gap_start - 1
                    while b >= 0 and self.positions[b] != (0, 0):
                        before_indices.insert(0, b)  # Insert at the beginning
                        b -= 1
                        if len(before_indices) == MIN_POINTS_FOR_EXTRAPOLATION:
                            break

                    if len(before_indices) >= 2:
                        times = before_indices
                        xs = [self.positions[idx][0] for idx in times]
                        ys = [self.positions[idx][1] for idx in times]

                        # Normalize time to start from zero
                        t0 = times[0]
                        t = [idx - t0 for idx in times]

                        # Fit quadratic functions to x and y over time
                        # For upward motion, y decreases in image coordinates
                        coeffs_x = np.polyfit(t, xs, 2)
                        coeffs_y = np.polyfit(t, ys, 2)

                        # Extrapolate positions during the gap
                        for k in range(gap_length):
                            gap_time = (gap_start + k) - t0
                            interp_x = int(np.polyval(coeffs_x, gap_time))
                            interp_y = int(np.polyval(coeffs_y, gap_time))
                            corrected_positions.append((interp_x, interp_y))
                    else:
                        # Not enough points to extrapolate, fill with zeros
                        corrected_positions.extend([(0, 0)] * gap_length)
                else:
                    # Gap too large, do not extrapolate
                    corrected_positions.extend([(0, 0)] * gap_length)

                i = gap_end
        return corrected_positions

    def process(self):
        # Proceed with the corrected positions
        corrected_positions = self.detect_teleportations()
        return corrected_positions

def load_processed_frames(preprocessed_csv_file):
    processed_frames = {}
    if os.path.exists(preprocessed_csv_file):
        with open(preprocessed_csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                frame_num = int(row['frame'])
                is_rally = int(row['is_rally_scene'] == 'True')
                processed_frames[frame_num] = is_rally
    else:
        print(f"Warning: {preprocessed_csv_file} does not exist. Assuming all frames are rally scenes.")
    return processed_frames

def process_video(video_file, model_file, num_frame, batch_size, save_dir):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_format = os.path.splitext(video_file)[1][1:]
    out_video_file = f'{save_dir}/{video_name}_shuttle.{video_format}'
    out_csv_file = f'{save_dir}/{video_name}_shuttle.csv'
    preprocessed_csv_file = f'{save_dir}/{video_name}_preprocessed.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    detector = Detector(model_file, num_frame, batch_size, device)

    os.makedirs(save_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*('DIVX' if video_format == 'avi' else 'mp4v'))

    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(out_video_file, fourcc, fps, (width, height))

    positions = []
    frame_buffer = []
    frame_count = 0
    csv_data = []

    processed_frames = load_processed_frames(preprocessed_csv_file)

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            pbar.update(1)

            is_rally = processed_frames.get(frame_count, 1) == 1  # Default to rally if not specified

            if is_rally:
                frame_buffer.append(frame)

                # Perform detection when enough frames are collected
                if len(frame_buffer) == num_frame * batch_size:
                    detected_positions = detector.detect(frame_buffer)
                    positions.extend(detected_positions)
                    frame_buffer = []
            else:
                # If not a rally frame, append (0, 0) for position
                positions.append((0, 0))

    # Process remaining frames in the buffer
    if frame_buffer:
        detected_positions = detector.detect(frame_buffer)
        positions.extend(detected_positions)

    post_processor = PostProcessor(positions)
    processed_positions = post_processor.process()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    with tqdm(total=total_frames, desc="Writing output video") as pbar:
        for pos in processed_positions:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            pbar.update(1)

            cx, cy = pos
            if cx != 0 or cy != 0:
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)
            csv_data.append({'Frame': frame_count, 'Visibility': 1 if cx != 0 or cy != 0 else 0, 'X': cx, 'Y': cy})

    # Save CSV file
    with open(out_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Frame', 'Visibility', 'X', 'Y'])
        writer.writeheader()
        writer.writerows(csv_data)

    out.release()
    cap.release()

def main(video_file):
    model_file = 'resources/model_best.pt'
    num_frame = 3
    batch_size = 8
    save_dir = 'result'

    process_video(video_file, model_file, num_frame, batch_size, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video to detect shuttle positions.')
    parser.add_argument('video_file', type=str, help='Path to the input video file.')
    args = parser.parse_args()
    main(args.video_file)