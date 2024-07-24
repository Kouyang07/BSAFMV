import os, cv2, argparse, numpy as np, torch, csv, math
from tqdm import tqdm
from collections import deque
from scipy.signal import savgol_filter

HEIGHT, WIDTH = 288, 512
BASE_TELEPORT_THRESHOLD = 50
DETECTION_THRESHOLD = 0.1
SMOOTHING_WINDOW = 15
MIN_TRAJECTORY_LENGTH = 5

def get_model(model_name, num_frame, input_type):
    from resources.model import TrackNetV2 as TrackNet
    return TrackNet(in_dim=num_frame * 3, out_dim=num_frame)

def get_frame_unit(frame_list, num_frame, device):
    batch = [np.array([cv2.resize(img, (WIDTH, HEIGHT)) for img in frame_list[i:i + num_frame]]).transpose((0, 3, 1, 2)).reshape(-1, HEIGHT, WIDTH) / 255.0 for i in range(0, len(frame_list), num_frame)]
    return torch.FloatTensor(np.array(batch)).to(device)

def get_object_center(heatmap):
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

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def calculate_adaptive_threshold(positions):
    if len(positions) < 2:
        return BASE_TELEPORT_THRESHOLD
    recent_speeds = [((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5 for p1, p2 in zip(positions[-10:], positions[-9:]) if p1[0] > 0 and p2[0] > 0]
    return max(BASE_TELEPORT_THRESHOLD, sum(recent_speeds) / len(recent_speeds) * 2) if recent_speeds else BASE_TELEPORT_THRESHOLD

def is_unrealistic_movement(prev_pos, curr_pos, positions):
    threshold = calculate_adaptive_threshold(positions)
    return prev_pos[0] > 0 and curr_pos[0] > 0 and ((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2) ** 0.5 > threshold

def load_processed_frames(processed_csv_file):
    with open(processed_csv_file, 'r') as csvfile:
        return {int(row['frame']): int(row['is_rally_scene'] == 'True') for row in csv.DictReader(csvfile)}

def detect_hits(positions, frame_shape, window_size=5, angle_threshold=30, velocity_threshold=2):
    hits = []
    for i in range(window_size, len(positions) - window_size):
        window = positions[i-window_size:i+window_size+1]
        if all(p[0] > 0 and p[1] > 0 for p in window):
            prev_vector = (window[window_size][0] - window[0][0], window[window_size][1] - window[0][1])
            next_vector = (window[-1][0] - window[window_size][0], window[-1][1] - window[window_size][1])

            prev_magnitude = math.sqrt(prev_vector[0]**2 + prev_vector[1]**2)
            next_magnitude = math.sqrt(next_vector[0]**2 + next_vector[1]**2)

            if prev_magnitude > velocity_threshold and next_magnitude > velocity_threshold:
                dot_product = prev_vector[0]*next_vector[0] + prev_vector[1]*next_vector[1]
                cos_angle = dot_product / (prev_magnitude * next_magnitude)
                angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))

                if angle > angle_threshold:
                    hits.append(i)
    return hits

def draw_hit_indicator(img, cx, cy, frame_number, hit_frame):
    if frame_number - hit_frame < 15:  # Show indicator for 15 frames
        radius = 40 - 2 * (frame_number - hit_frame)  # Shrinking circle
        opacity = 1 - (frame_number - hit_frame) / 15  # Fading opacity
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), radius, (0, 255, 0), 2)
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

        # Add "HIT!" text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("HIT!", font, 1, 2)[0]
        text_x = cx - text_size[0] // 2
        text_y = cy - radius - 10
        cv2.putText(img, "HIT!", (text_x, text_y), font, 1, (0, 255, 0), 2)

def smooth_trajectory(positions):
    x, y = zip(*positions)
    return list(zip(savgol_filter(x, SMOOTHING_WINDOW, 3).astype(int), savgol_filter(y, SMOOTHING_WINDOW, 3).astype(int)))

def process_frame(cx_pred, cy_pred, positions, frame_shape):
    h, w = frame_shape[:2]
    if 0 < cx_pred < w and 0 < cy_pred < h:
        if positions and is_unrealistic_movement(positions[-1], (cx_pred, cy_pred), list(positions)):
            cx_pred, cy_pred = 0, 0
    else:
        cx_pred, cy_pred = 0, 0
    return cx_pred, cy_pred, int(cx_pred > 0 and cy_pred > 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file', type=str, help='Path to the video file')
    args = parser.parse_args()

    video_file = args.video_file
    model_file = 'resources/model_best.pt'
    num_frame, batch_size = 3, 8
    save_dir = 'result'

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_format = os.path.splitext(video_file)[1][1:]
    out_video_file = f'{save_dir}/{video_name}_shuttle.{video_format}'
    out_csv_file = f'{save_dir}/{video_name}_shuttle.csv'
    preprocessed_csv_file = f'{save_dir}/{video_name}_preprocessed.csv'

    device = get_device()
    checkpoint = torch.load(model_file, map_location=device)
    param_dict = checkpoint['param_dict']
    model = get_model(param_dict['model_name'], param_dict['num_frame'], param_dict['input_type']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*('DIVX' if video_format == 'avi' else 'mp4v'))
    processed_frames = load_processed_frames(preprocessed_csv_file)

    cap = cv2.VideoCapture(video_file)
    fps, w, h = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = h / HEIGHT
    out = cv2.VideoWriter(out_video_file, fourcc, fps, (w, h))

    positions = []
    frame_count, rally_frame_buffer, hits = 0, [], []
    last_hit_frame = 0

    with open(out_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Frame', 'Visibility', 'X', 'Y', 'Hit'])
        writer.writeheader()

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_count += 1
                pbar.update(1)

                if frame_count in processed_frames and processed_frames[frame_count] == 1:
                    rally_frame_buffer.append(frame)
                    if len(rally_frame_buffer) == num_frame * batch_size:
                        x = get_frame_unit(rally_frame_buffer, num_frame, device)
                        with torch.no_grad():
                            y_pred = model(x)
                        h_pred = (y_pred.detach().cpu().numpy() > 0.5).astype('uint8') * 255
                        h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

                        for i, img in enumerate(rally_frame_buffer):
                            cx_pred, cy_pred = get_object_center(h_pred[i])
                            cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)

                            cx_pred, cy_pred, vis = process_frame(cx_pred, cy_pred, positions, img.shape)

                            positions.append((cx_pred, cy_pred))
                            current_frame = frame_count - len(rally_frame_buffer) + i + 1

                            if len(positions) > 15:  # Increased window size for hit detection
                                new_hits = detect_hits(positions[-15:], img.shape)
                                if new_hits and current_frame - last_hit_frame > 10:  # Reduced minimum frames between hits
                                    hits.append(current_frame)
                                    last_hit_frame = current_frame
                                    print(f"Hit detected at frame {current_frame}")  # Debug print

                            is_hit = int(current_frame in hits)
                            writer.writerow({'Frame': current_frame, 'Visibility': vis, 'X': cx_pred, 'Y': cy_pred, 'Hit': is_hit})

                            if cx_pred != 0 or cy_pred != 0:
                                cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                            if is_hit or current_frame - last_hit_frame < 15:
                                draw_hit_indicator(img, cx_pred, cy_pred, current_frame, last_hit_frame)

                            # Debug visualization
                            cv2.putText(img, f"Frame: {current_frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(img, f"Hits: {len(hits)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            out.write(img)
                        rally_frame_buffer = []
                else:
                    if rally_frame_buffer:
                        for img in rally_frame_buffer:
                            out.write(img)
                        rally_frame_buffer = []
                    writer.writerow({'Frame': frame_count, 'Visibility': 0, 'X': 0, 'Y': 0, 'Hit': 0})
                    out.write(frame)
                    positions.append((0, 0))

            if rally_frame_buffer:
                x = get_frame_unit(rally_frame_buffer, num_frame, device)
                with torch.no_grad():
                    y_pred = model(x)
                h_pred = (y_pred.detach().cpu().numpy() > 0.5).astype('uint8') * 255
                h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

                for i, img in enumerate(rally_frame_buffer):
                    cx_pred, cy_pred = get_object_center(h_pred[i])
                    cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)

                    cx_pred, cy_pred, vis = process_frame(cx_pred, cy_pred, positions, img.shape)

                    positions.append((cx_pred, cy_pred))
                    current_frame = frame_count - len(rally_frame_buffer) + i + 1

                    if len(positions) > MIN_TRAJECTORY_LENGTH:
                        new_hits = detect_hits(list(positions)[-MIN_TRAJECTORY_LENGTH:], img.shape)
                        if new_hits:
                            hits.append(current_frame - MIN_TRAJECTORY_LENGTH + new_hits[-1])

                    is_hit = int(current_frame in hits)
                    writer.writerow({'Frame': current_frame, 'Visibility': vis, 'X': cx_pred, 'Y': cy_pred, 'Hit': is_hit})
                    if cx_pred != 0 or cy_pred != 0:
                        cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                    if is_hit:
                        cv2.putText(img, "HIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    out.write(img)

    with open(out_csv_file, 'r') as csvfile:
        rows = list(csv.DictReader(csvfile))

    positions = [(int(row['X']), int(row['Y'])) for row in rows]
    smoothed_positions = smooth_trajectory(positions)

    with open(out_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Frame', 'Visibility', 'X', 'Y', 'Hit'])
        writer.writeheader()
        for row, smoothed_pos in zip(rows, smoothed_positions):
            row['X'], row['Y'] = smoothed_pos
            writer.writerow(row)

    out.release()
    print(f"Total hits detected: {len(hits)}")
    print(f"Hit frames: {hits}")
    print('Done.')

if __name__ == "__main__":
    main()
