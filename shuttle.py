import logging
import os, cv2, argparse, numpy as np, torch, csv
from tqdm import tqdm

HEIGHT, WIDTH = 288, 512
DETECTION_THRESHOLD, SMOOTHING_WINDOW = 0.05, 11
STATIONARY_THRESHOLD, STATIONARY_FRAMES = 10, 10
BASE_TELEPORT_THRESHOLD, MAX_ESTIMATION_FRAMES = 60, 5

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
        batch = [np.array([cv2.resize(img, (WIDTH, HEIGHT)) for img in frame_list[i:i + self.num_frame]]).transpose(
            (0, 3, 1, 2)).reshape(-1, HEIGHT, WIDTH) / 255.0 for i in range(0, len(frame_list), self.num_frame)]
        return torch.FloatTensor(np.array(batch)).to(self.device)

    def get_object_center(self, heatmap):
        if np.amax(heatmap) < DETECTION_THRESHOLD:
            return 0, 0
        binary = cv2.threshold(heatmap, DETECTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return 0, 0
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    def detect(self, frame_buffer, is_rally):
        x = self.get_frame_unit(frame_buffer)
        with torch.no_grad():
            y_pred = self.model(x)
        h_pred = (y_pred.detach().cpu().numpy() > 0.5).astype('uint8') * 255
        h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

        positions = []
        for i, img in enumerate(frame_buffer):
            cx_pred, cy_pred = self.get_object_center(h_pred[i])
            if is_rally:
                cx_pred, cy_pred = int(img.shape[0] / HEIGHT * cx_pred), int(img.shape[1] / WIDTH * cy_pred)
            else:
                cx_pred, cy_pred = 0, 0
            positions.append((cx_pred, cy_pred))

        return positions


class PostProcessor:
    def __init__(self, positions):
        self.positions = positions

    def smooth_trajectory(self):
        smoothed_positions = []
        window_size = 10
        for i in range(len(self.positions)):
            window = self.positions[max(0, i - window_size // 2):min(len(self.positions), i + window_size // 2 + 1)]
            x = sum(p[0] for p in window) / len(window)
            y = sum(p[1] for p in window) / len(window)
            smoothed_positions.append((x, y))
        return smoothed_positions

    def remove_stationary_segments(self):
        filtered_positions = []
        for i in range(len(self.positions)):
            if self.positions[i][0] != 0 or self.positions[i][1] != 0:
                filtered_positions.append(self.positions[i])
            elif i > 0 and i < len(self.positions) - 1:
                if self.positions[i-1][0] != 0 and self.positions[i+1][0] != 0:
                    filtered_positions.append(self.estimate_position(self.positions[i-1], self.positions[i+1]))
                else:
                    filtered_positions.append((0, 0))
            else:
                filtered_positions.append((0, 0))
        return filtered_positions

    def estimate_position(self, prev_pos, next_pos):
        return (prev_pos[0] + next_pos[0]) // 2, (prev_pos[1] + next_pos[1]) // 2

    def remove_outliers(self):
        filtered_positions = []
        for i in range(len(self.positions)):
            if self.positions[i][0] != 0 or self.positions[i][1] != 0:
                if i > 0 and i < len(self.positions) - 1:
                    if abs(self.positions[i][0] - self.positions[i-1][0]) > 50 or abs(self.positions[i][0] - self.positions[i+1][0]) > 50:
                        filtered_positions.append((0, 0))
                    else:
                        filtered_positions.append(self.positions[i])
                else:
                    filtered_positions.append(self.positions[i])
            else:
                filtered_positions.append((0, 0))
        return filtered_positions

    def process(self):
        self.positions = self.smooth_trajectory()
        self.positions = self.remove_stationary_segments()
        self.positions = self.remove_outliers()
        return self.positions

def load_processed_frames(processed_csv_file):
    with open(processed_csv_file, 'r') as csvfile:
        return {int(row['frame']): int(row['is_rally_scene'] == 'True') for row in csv.DictReader(csvfile)}

def process_video(video_file, model_file, num_frame, batch_size, save_dir):
    logging.info(f"Processing video: {video_file}")
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_format = os.path.splitext(video_file)[1][1:]
    out_video_file = f'{save_dir}/{video_name}_shuttle.{video_format}'
    out_csv_file = f'{save_dir}/{video_name}_shuttle.csv'
    preprocessed_csv_file = f'{save_dir}/{video_name}_preprocessed.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    detector = Detector(model_file, num_frame, batch_size, device)

    os.makedirs(save_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*('DIVX' if video_format == 'avi' else 'mp4v'))
    processed_frames = load_processed_frames(preprocessed_csv_file)

    cap = cv2.VideoCapture(video_file)
    fps, w, h = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(out_video_file, fourcc, fps, (w, h))

    positions = []
    frame_count = 0
    rally_frame_buffer = []
    csv_data = []

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            pbar.update(1)

            is_rally = processed_frames.get(frame_count, 0) == 1

            if is_rally:
                rally_frame_buffer.append(frame)
                if len(rally_frame_buffer) == num_frame * batch_size:
                    detected_positions = detector.detect(rally_frame_buffer, is_rally)

                    for i, img in enumerate(rally_frame_buffer):
                        cx_pred, cy_pred = detected_positions[i]
                        cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                        out.write(img)
                        csv_data.append({'Frame': frame_count - len(rally_frame_buffer) + i, 'Visibility': 1 if cx_pred != 0 or cy_pred != 0 else 0, 'X': cx_pred, 'Y': cy_pred})
                    rally_frame_buffer = []
            else:
                if rally_frame_buffer:
                    for i, _ in enumerate(rally_frame_buffer):
                        out.write(frame)
                        csv_data.append({'Frame': frame_count - len(rally_frame_buffer) + i, 'Visibility': 0, 'X': 0, 'Y': 0})
                    rally_frame_buffer = []
                out.write(frame)
                csv_data.append({'Frame': frame_count, 'Visibility': 0, 'X': 0, 'Y': 0})

    # Write the CSV file with the data used for annotation
    with open(out_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Frame', 'Visibility', 'X', 'Y'])
        writer.writeheader()
        writer.writerows(csv_data)

    out.release()
    logging.info('Processing completed.')

def main(video_file):
    model_file, num_frame, batch_size = 'resources/model_best.pt', 3, 8
    save_dir = 'result'

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    process_video(video_file, model_file, num_frame, batch_size, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video to detect shuttle positions.')
    parser.add_argument('video_file', type=str, help='Path to the input video file.')
    args = parser.parse_args()
    main(args.video_file)