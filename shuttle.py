import os, cv2, argparse, numpy as np, torch, csv
from tqdm import tqdm

HEIGHT, WIDTH = 288, 512
TELEPORT_THRESHOLD = 50
MAX_INTERPOLATE_FRAMES = 15

def get_model(model_name, num_frame, input_type):
    from resources.model import TrackNetV2 as TrackNet
    return TrackNet(in_dim=num_frame * 3, out_dim=num_frame)

def get_frame_unit(frame_list, num_frame, device):
    batch = []
    for i in range(0, len(frame_list), num_frame):
        frames = np.array([cv2.resize(img, (WIDTH, HEIGHT)) for img in frame_list[i:i + num_frame]])
        frames = frames.transpose((0, 3, 1, 2)).reshape(-1, HEIGHT, WIDTH) / 255.0
        batch.append(frames)
    return torch.FloatTensor(np.array(batch)).to(device)

def get_object_center(heatmap):
    if np.amax(heatmap) == 0:
        return 0, 0
    (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    target = max(rects, key=lambda r: r[2] * r[3])
    return int(target[0] + target[2] / 2), int(target[1] + target[3] / 2)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def is_unrealistic_movement(prev_pos, curr_pos, threshold=TELEPORT_THRESHOLD):
    return prev_pos[0] > 0 and curr_pos[0] > 0 and abs(curr_pos[0] - prev_pos[0]) > threshold

def load_processed_frames(processed_csv_file):
    with open(processed_csv_file, 'r') as csvfile:
        return {int(row['frame']): int(row['is_rally_scene'] == 'True') for row in csv.DictReader(csvfile)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file', type=str, help='Path to the video file')
    args = parser.parse_args()

    video_file = args.video_file
    model_file = 'resources/model_best.pt'
    num_frame = 3
    batch_size = 8
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

    with open(out_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Frame', 'Visibility', 'X', 'Y'])
        writer.writeheader()

        positions = []
        frame_count = 0
        rally_frame_buffer = []
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

                            if positions and is_unrealistic_movement(positions[-1], (cx_pred, cy_pred)):
                                cx_pred, cy_pred = 0, 0

                            vis = int(cx_pred > 0 and cy_pred > 0)
                            positions.append((cx_pred, cy_pred))
                            writer.writerow({'Frame': frame_count - len(rally_frame_buffer) + i + 1, 'Visibility': vis, 'X': cx_pred, 'Y': cy_pred})
                            if cx_pred != 0 or cy_pred != 0:
                                cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                            out.write(img)
                        rally_frame_buffer = []
                else:
                    if rally_frame_buffer:
                        for img in rally_frame_buffer:
                            out.write(img)
                        rally_frame_buffer = []
                    writer.writerow({'Frame': frame_count, 'Visibility': 0, 'X': 0, 'Y': 0})
                    out.write(frame)

            # Process any remaining frames in the buffer
            if rally_frame_buffer:
                x = get_frame_unit(rally_frame_buffer, num_frame, device)
                with torch.no_grad():
                    y_pred = model(x)
                h_pred = (y_pred.detach().cpu().numpy() > 0.5).astype('uint8') * 255
                h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

                for i, img in enumerate(rally_frame_buffer):
                    cx_pred, cy_pred = get_object_center(h_pred[i])
                    cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)

                    if positions and is_unrealistic_movement(positions[-1], (cx_pred, cy_pred)):
                        cx_pred, cy_pred = 0, 0

                    vis = int(cx_pred > 0 and cy_pred > 0)
                    positions.append((cx_pred, cy_pred))
                    writer.writerow({'Frame': frame_count - len(rally_frame_buffer) + i + 1, 'Visibility': vis, 'X': cx_pred, 'Y': cy_pred})
                    if cx_pred != 0 or cy_pred != 0:
                        cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                    out.write(img)

    out.release()
    print('Done.')

if __name__ == "__main__":
    main()
