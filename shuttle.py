import os
import csv
from typing import List, Tuple, Dict
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

HEIGHT = 288
WIDTH = 512
TELEPORT_THRESHOLD = 50
MAX_INTERPOLATE_FRAMES = 15

def get_model(model_name: str, num_frame: int, input_type: str) -> torch.nn.Module:
    """Create model by name and configuration parameter."""
    if model_name == 'TrackNetV2':
        from resources.model import TrackNetV2 as TrackNet
        return TrackNet(in_dim=num_frame*3, out_dim=num_frame)
    raise ValueError(f"Unsupported model: {model_name}")

def get_frame_unit(frame_list: List[np.ndarray], num_frame: int, device: torch.device) -> torch.Tensor:
    """Sample frames from the video."""
    def get_unit(frames: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([cv2.resize(img, (WIDTH, HEIGHT))[np.newaxis, ...] for img in frames], axis=0)

    batch = []
    for i in range(0, len(frame_list), num_frame):
        frames = get_unit(frame_list[i: i+num_frame])
        frames = frames.transpose((0, 3, 1, 2)).reshape(-1, HEIGHT, WIDTH) / 255.0
        batch.append(frames)

    return torch.FloatTensor(np.array(batch)).to(device)

def get_object_center(heatmap: np.ndarray) -> Tuple[int, int]:
    """Get coordinates from the heatmap."""
    if np.amax(heatmap) == 0:
        return 0, 0

    (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]

    max_area_idx = max(range(len(rects)), key=lambda i: rects[i][2] * rects[i][3])
    target = rects[max_area_idx]

    return int(target[0] + target[2] / 2), int(target[1] + target[3] / 2)

def get_device() -> torch.device:
    """Returns the best available device (CUDA if available, otherwise MPS, then CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def is_unrealistic_movement(prev_pos: Tuple[int, int], curr_pos: Tuple[int, int], threshold: int = TELEPORT_THRESHOLD) -> bool:
    """Check for unrealistic horizontal shuttle movement."""
    return prev_pos[0] > 0 and curr_pos[0] > 0 and abs(curr_pos[0] - prev_pos[0]) > threshold

def get_direction(prev_pos: Tuple[int, int], curr_pos: Tuple[int, int]) -> int:
    """Determine the direction of the shuttle movement."""
    if prev_pos[1] == 0 or curr_pos[1] == 0:
        return 1  # Default to downward if we don't have enough information
    return 1 if curr_pos[1] > prev_pos[1] else 0

def process_video(video_file: str, model_file: str, save_dir: str):
    """Main function to process the video and track the shuttle."""
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_format = os.path.splitext(video_file)[1][1:]
    out_video_file = f'{save_dir}/{video_name}_shuttle.{video_format}'
    out_csv_file = f'{save_dir}/{video_name}_shuttle.csv'

    device = get_device()
    print(f'Using device: {device}')

    checkpoint = torch.load(model_file, map_location=device)
    param_dict = checkpoint['param_dict']
    model_name = param_dict['model_name']
    num_frame = param_dict['num_frame']
    input_type = param_dict['input_type']
    batch_size = 8

    os.makedirs(save_dir, exist_ok=True)

    model = get_model(model_name, num_frame, input_type).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'DIVX' if video_format == 'avi' else 'mp4v')

    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = h / HEIGHT

    out = cv2.VideoWriter(out_video_file, fourcc, fps, (w, h))

    with open(out_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Frame', 'Visibility', 'X', 'Y', 'Direction'])
        writer.writeheader()

        positions = []
        prev_position = (0, 0)
        frame_count = 0

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                frame_queue = []
                for _ in range(num_frame * batch_size):
                    success, frame = cap.read()
                    if not success:
                        break
                    frame_count += 1
                    frame_queue.append(frame)
                    pbar.update(1)

                if not frame_queue:
                    break

                x = get_frame_unit(frame_queue, num_frame, device)

                with torch.no_grad():
                    y_pred = model(x)

                h_pred = (y_pred.detach().cpu().numpy() > 0.5).astype('uint8') * 255
                h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

                for i, img in enumerate(frame_queue):
                    frame_index = frame_count - len(frame_queue) + i
                    cx_pred, cy_pred = get_object_center(h_pred[i])
                    cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)

                    if positions and is_unrealistic_movement(positions[-1], (cx_pred, cy_pred)):
                        cx_pred, cy_pred = 0, 0

                    vis = 1 if cx_pred > 0 and cy_pred > 0 else 0
                    direction = get_direction(prev_position, (cx_pred, cy_pred))
                    positions.append((cx_pred, cy_pred))
                    writer.writerow({
                        'Frame': frame_index,
                        'Visibility': vis,
                        'X': cx_pred,
                        'Y': cy_pred,
                        'Direction': direction
                    })

                    if cx_pred != 0 or cy_pred != 0:
                        # Use blue for upward movement, red for downward movement
                        color = (255, 0, 0) if direction == 0 else (0, 0, 255)
                        cv2.circle(img, (cx_pred, cy_pred), 5, color, -1)

                    prev_position = (cx_pred, cy_pred)
                    out.write(img)

    out.release()
    print('Processing completed.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track badminton shuttle in video.")
    parser.add_argument('video_file', type=str, help='Path to the video file')
    parser.add_argument('--model_file', type=str, default='resources/model_best.pt', help='Path to the model file')
    parser.add_argument('--save_dir', type=str, default='result', help='Directory to save output')

    args = parser.parse_args()

    process_video(args.video_file, args.model_file, args.save_dir)
