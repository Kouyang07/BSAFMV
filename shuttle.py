import os, cv2, argparse, numpy as np, torch, csv
from tqdm import tqdm
from scipy.signal import savgol_filter
from collections import deque

HEIGHT, WIDTH = 288, 512
DETECTION_THRESHOLD, SMOOTHING_WINDOW = 0.05, 11
STATIONARY_THRESHOLD, STATIONARY_FRAMES = 10, 2
BASE_TELEPORT_THRESHOLD, MAX_ESTIMATION_FRAMES = 60, 5

def get_model(model_name, num_frame, input_type):
    from resources.model import TrackNetV2 as TrackNet
    return TrackNet(in_dim=num_frame * 3, out_dim=num_frame)

def get_frame_unit(frame_list, num_frame, device):
    batch = [np.array([cv2.resize(img, (WIDTH, HEIGHT)) for img in frame_list[i:i + num_frame]]).transpose(
        (0, 3, 1, 2)).reshape(-1, HEIGHT, WIDTH) / 255.0 for i in range(0, len(frame_list), num_frame)]
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
    recent_positions = list(positions)[-10:]
    recent_speeds = [((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
                     for p1, p2 in zip(recent_positions[:-1], recent_positions[1:])
                     if p1[0] > 0 and p2[0] > 0]
    return max(BASE_TELEPORT_THRESHOLD, sum(recent_speeds) / len(recent_speeds) * 2) if recent_speeds else BASE_TELEPORT_THRESHOLD

def is_unrealistic_movement(prev_pos, curr_pos, positions):
    threshold = calculate_adaptive_threshold(positions)
    return prev_pos[0] > 0 and curr_pos[0] > 0 and ((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2) ** 0.5 > threshold

def load_processed_frames(processed_csv_file):
    with open(processed_csv_file, 'r') as csvfile:
        return {int(row['frame']): int(row['is_rally_scene'] == 'True') for row in csv.DictReader(csvfile)}

def is_stationary(positions, window_size=STATIONARY_FRAMES):
    if len(positions) < window_size:
        return False
    recent_positions = positions[-window_size:]
    if any(p[0] == 0 and p[1] == 0 for p in recent_positions):
        return False
    max_distance = max(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 for p1, p2 in zip(recent_positions, recent_positions[1:]))
    return max_distance < STATIONARY_THRESHOLD

def smooth_trajectory(positions):
    x, y = zip(*positions)
    return list(zip(savgol_filter(x, SMOOTHING_WINDOW, 3).astype(int), savgol_filter(y, SMOOTHING_WINDOW, 3).astype(int)))

def estimate_position(prev_pos, next_pos):
    return (prev_pos[0] + next_pos[0]) // 2, (prev_pos[1] + next_pos[1]) // 2

def process_frame(cx_pred, cy_pred, positions, frame_shape, is_rally):
    h, w = frame_shape[:2]
    if is_rally:
        if cx_pred == 0 or cy_pred == 0:
            if len(positions) >= 2:
                cx_pred, cy_pred = estimate_position(positions[-1], positions[-2])
            vis = 2
        else:
            vis = 1
    else:
        vis = 0

    if 0 < cx_pred < w and 0 < cy_pred < h:
        if positions and is_unrealistic_movement(list(positions)[-1], (cx_pred, cy_pred), positions):
            cx_pred, cy_pred, vis = 0, 0, 0
        elif is_stationary(list(positions) + [(cx_pred, cy_pred)]):
            cx_pred, cy_pred, vis = 0, 0, 0
    else:
        cx_pred, cy_pred, vis = 0, 0, 0

    return cx_pred, cy_pred, vis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file', type=str, help='Path to the video file')
    args = parser.parse_args()

    video_file = args.video_file
    model_file, num_frame, batch_size = 'resources/model_best.pt', 3, 8
    save_dir = 'result'

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_format = os.path.splitext(video_file)[1][1:]
    out_video_file = f'{save_dir}/{video_name}_shuttle.{video_format}'
    out_csv_file = f'{save_dir}/{video_name}_shuttle.csv'
    preprocessed_csv_file = f'{save_dir}/{video_name}_preprocessed.csv'

    device = get_device()
    checkpoint = torch.load(model_file, map_location=device)
    model = get_model(checkpoint['param_dict']['model_name'], checkpoint['param_dict']['num_frame'], checkpoint['param_dict']['input_type']).to(device)
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

    positions = deque(maxlen=total_frames)
    frame_count, rally_frame_buffer = 0, []

    with open(out_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Frame', 'Visibility', 'X', 'Y'])
        writer.writeheader()

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
                        x = get_frame_unit(rally_frame_buffer, num_frame, device)
                        with torch.no_grad():
                            y_pred = model(x)
                        h_pred = (y_pred.detach().cpu().numpy() > 0.5).astype('uint8') * 255
                        h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

                        for i, img in enumerate(rally_frame_buffer):
                            try:
                                cx_pred, cy_pred = get_object_center(h_pred[i])
                                cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)

                                cx_pred, cy_pred, vis = process_frame(cx_pred, cy_pred, positions, img.shape, is_rally)

                                positions.append((cx_pred, cy_pred))

                                writer.writerow({'Frame': frame_count - len(rally_frame_buffer) + i + 1, 'Visibility': vis, 'X': cx_pred, 'Y': cy_pred})

                                if vis > 0:  # Draw for both visibility 1 and 2
                                    color = (0, 0, 255) if vis == 1 else (0, 255, 255)
                                    cv2.circle(img, (cx_pred, cy_pred), 5, color, -1)

                                out.write(img)
                            except Exception as e:
                                print(f"Error processing frame {frame_count - len(rally_frame_buffer) + i + 1}: {str(e)}")
                                positions.append((0, 0))
                                writer.writerow({'Frame': frame_count - len(rally_frame_buffer) + i + 1, 'Visibility': 0, 'X': 0, 'Y': 0})
                                out.write(img)
                        rally_frame_buffer = []
                else:
                    if rally_frame_buffer:
                        for img in rally_frame_buffer:
                            out.write(img)
                        rally_frame_buffer = []
                    writer.writerow({'Frame': frame_count, 'Visibility': 0, 'X': 0, 'Y': 0})
                    out.write(frame)
                    positions.append((0, 0))

    # Post-processing
    smoothed_positions = smooth_trajectory(list(positions))

    with open(out_csv_file, 'r') as csvfile:
        rows = list(csv.DictReader(csvfile))

    with open(out_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Frame', 'Visibility', 'X', 'Y'])
        writer.writeheader()
        for i, (row, smoothed_pos) in enumerate(zip(rows, smoothed_positions)):
            if int(row['Visibility']) == 0 and i > 0 and i < len(rows) - 1:
                if int(rows[i-1]['Visibility']) > 0 and int(rows[i+1]['Visibility']) > 0:
                    row['X'], row['Y'] = estimate_position((int(rows[i-1]['X']), int(rows[i-1]['Y'])),
                                                           (int(rows[i+1]['X']), int(rows[i+1]['Y'])))
                    row['Visibility'] = '2'
            elif int(row['Visibility']) > 0:
                row['X'], row['Y'] = smoothed_pos
            writer.writerow(row)

    out.release()
    print('Done.')

if __name__ == "__main__":
    main()
