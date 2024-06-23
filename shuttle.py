import os
import cv2
import argparse
import numpy as np
import torch
from tqdm import tqdm
import csv

HEIGHT = 288
WIDTH = 512
TELEPORT_THRESHOLD = 50  # Adjust this value based on the expected max shuttle speed between frames
MAX_INTERPOLATE_FRAMES = 15  # Maximum number of frames to interpolate

def get_model(model_name, num_frame, input_type):
    """Create model by name and configuration parameter."""
    if model_name == 'TrackNetV2':
        from resources.model import TrackNetV2 as TrackNet

    if model_name in ['TrackNetV2']:
        model = TrackNet(in_dim=num_frame*3, out_dim=num_frame)

    return model

def get_frame_unit(frame_list, num_frame, device):
    """Sample frames from the video."""
    batch = []
    h, w, _ = frame_list[0].shape
    h_ratio = h / HEIGHT
    w_ratio = w / WIDTH

    def get_unit(frame_list):
        """Generate an input sequence from frames."""
        frames = np.array([]).reshape(0, HEIGHT, WIDTH, 3)

        for img in frame_list:
            img = cv2.resize(img, (WIDTH, HEIGHT))
            frames = np.concatenate((frames, img[np.newaxis, ...]), axis=0)

        return frames

    for i in range(0, len(frame_list), num_frame):
        frames = get_unit(frame_list[i: i+num_frame])
        frames = frames.transpose((0, 3, 1, 2))  # Change shape to (F, 3, H, W)
        frames = frames.reshape(-1, HEIGHT, WIDTH)
        frames = frames / 255.0
        batch.append(frames)

    batch = np.array(batch)
    return torch.FloatTensor(batch).to(device)

def get_object_center(heatmap):
    """Get coordinates from the heatmap."""
    if np.amax(heatmap) == 0:
        return 0, 0
    else:
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        max_area_idx = 0
        max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
        for i in range(len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        target = rects[max_area_idx]

    return int((target[0] + target[2] / 2)), int((target[1] + target[3] / 2))

def get_device():
    """Returns the best available device (CUDA if available, otherwise MPS, then CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def is_unrealistic_movement(prev_pos, curr_pos, threshold=TELEPORT_THRESHOLD):
    """Check for unrealistic horizontal shuttle movement."""
    if prev_pos[0] > 0 and curr_pos[0] > 0:
        return abs(curr_pos[0] - prev_pos[0]) > threshold
    return False

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str)
parser.add_argument('--model_file', type=str, default='resources/model_best.pt')
parser.add_argument('--num_frame', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--save_dir', type=str, default='result')
args = parser.parse_args()

video_file = args.video_file
model_file = args.model_file
num_frame = args.num_frame
batch_size = args.batch_size
save_dir = args.save_dir

video_name = video_file.split('/')[-1][:-4]
video_format = video_file.split('/')[-1][-3:]
out_video_file = f'{save_dir}/{video_name}_pred.{video_format}'
out_csv_file = f'{save_dir}/{video_name}_ball.csv'

device = get_device()

checkpoint = torch.load(model_file, map_location=torch.device(device))
print('Using device:', device)
param_dict = checkpoint['param_dict']
model_name = param_dict['model_name']
num_frame = param_dict['num_frame']
input_type = param_dict['input_type']

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = get_model(model_name, num_frame, input_type).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

if video_format == 'avi':
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
elif video_format == 'mp4':
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
    raise ValueError('Invalid video format.')

with open(out_csv_file, 'w', newline='') as csvfile:
    fieldnames = ['Frame', 'Visibility', 'X', 'Y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    frame_count = 0
    num_final_frame = 0
    ratio = h / HEIGHT
    out = cv2.VideoWriter(out_video_file, fourcc, fps, (w, h))

    positions = []

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while success:
            frame_queue = []
            for _ in range(num_frame * batch_size):
                success, frame = cap.read()
                if not success:
                    break
                else:
                    frame_count += 1
                    frame_queue.append(frame)
                    pbar.update(1)

            if not frame_queue:
                break

            if len(frame_queue) % num_frame != 0:
                frame_queue = []
                num_final_frame = len(frame_queue) + 1
                frame_count = frame_count - num_frame * batch_size
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                for _ in range(num_frame * batch_size):
                    success, frame = cap.read()
                    if not success:
                        break
                    else:
                        frame_count += 1
                        frame_queue.append(frame)
                        pbar.update(1)
                if len(frame_queue) % num_frame != 0:
                    continue

            x = get_frame_unit(frame_queue, num_frame, device)

            with torch.no_grad():
                y_pred = model(x)
            y_pred = y_pred.detach().cpu().numpy()
            h_pred = y_pred > 0.5
            h_pred = h_pred * 255.
            h_pred = h_pred.astype('uint8')
            h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

            for i in range(h_pred.shape[0]):
                if num_final_frame > 0 and i < (num_frame * batch_size - num_final_frame - 1):
                    continue
                else:
                    if i >= len(frame_queue):
                        break

                    img = frame_queue[i].copy()
                    cx_pred, cy_pred = get_object_center(h_pred[i])
                    cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)

                    if len(positions) > 0 and is_unrealistic_movement(positions[-1], (cx_pred, cy_pred)):
                        cx_pred, cy_pred = 0, 0  # Mark as invalid if unrealistic movement is detected

                    vis = 1 if cx_pred > 0 and cy_pred > 0 else 0
                    positions.append((cx_pred, cy_pred))
                    writer.writerow({'Frame': frame_count - (num_frame * batch_size) + i, 'Visibility': vis, 'X': cx_pred, 'Y': cy_pred})
                    if cx_pred != 0 or cy_pred != 0:
                        cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                    out.write(img)

out.release()
print('Done.')