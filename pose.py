import cv2
import torch
import torchvision
import numpy as np
import copy
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import sys
import csv
import os
from tqdm import tqdm

class PoseDetect:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_RCNN()

    def reset(self):
        self.got_info = False

    def setup_RCNN(self):
        self.__pose_kpRCNN = torch.load('resources/pose_kpRCNN.pth')
        self.__pose_kpRCNN.to(self.device).eval()

    def del_RCNN(self):
        del self.__pose_kpRCNN

    def get_human_joints(self, frame):
        frame_copy = frame.copy()
        outputs = self.__human_detection(frame_copy)
        human_joints = outputs[0]['keypoints'].cpu().detach().numpy()
        filtered_outputs = []
        for i in range(len(human_joints)):
            filtered_outputs.append(human_joints[i].tolist())

        for points in filtered_outputs:
            for i, joints in enumerate(points):
                points[i] = joints[0:2]
        filtered_outputs
        return outputs, filtered_outputs

    def __human_detection(self, frame):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        t_image = transforms.Compose(
            [transforms.ToTensor()])(pil_image).unsqueeze(0).to(self.device)
        outputs = self.__pose_kpRCNN(t_image)
        return outputs

    def draw_key_points(self, filtered_outputs, image, human_limit=-1):
        image_copy = image.copy()
        edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12),
                 (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
                 (12, 14), (14, 16), (5, 6)]

        # top player is blue and bottom one is red
        top_color_edge = (255, 0, 0)
        bot_color_edge = (0, 0, 255)
        top_color_joint = (115, 47, 14)
        bot_color_joint = (35, 47, 204)


        for i in range(len(filtered_outputs)):

            if i > human_limit and human_limit != -1:
                break

            color = top_color_edge if i == 0 else bot_color_edge
            color_joint = top_color_joint if i == 0 else bot_color_joint

            keypoints = np.array(filtered_outputs[i])  # 17, 2
            keypoints = keypoints[:, :].reshape(-1, 2)
            for p in range(keypoints.shape[0]):
                cv2.circle(image_copy,
                           (int(keypoints[p, 0]), int(keypoints[p, 1])),
                           3,
                           color_joint,
                           thickness=-1,
                           lineType=cv2.FILLED)

            for e in edges:
                cv2.line(image_copy,
                         (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                         (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                         color,
                         2,
                         lineType=cv2.LINE_AA)
        return image_copy

def load_processed_frames(processed_csv_file):
    with open(processed_csv_file, 'r') as csvfile:
        return {int(row['frame']): int(row['is_rally_scene'] == 'True') for row in csv.DictReader(csvfile)}

def read_court_coordinates(file_path):
    print(f"Reading court coordinates from {file_path}")
    with open(file_path, 'r') as file:
        coordinates = [tuple(map(float, line.strip().split(';'))) for line in file.readlines()[:4]]
    print(f"Court coordinates: {coordinates}")
    return coordinates

def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def box_intersects_court(box, court_coords):
    x1, y1, x2, y2 = box
    return any(point_inside_polygon(x, y, court_coords) for x, y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)])

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the base name of the video file
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create the result directory if it doesn't exist
    os.makedirs("result", exist_ok=True)

    # Prepare the CSV file
    csv_path = f"result/{base_name}_pose.csv"

    # Prepare the output video
    out_path = f"result/{base_name}_pose.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Load processed frames
    preprocessed_path = f"result/{base_name}_preprocessed.csv"
    rally_frames = load_processed_frames(preprocessed_path)

    # Load court coordinates
    court_path = f"result/{base_name}_court.txt"
    court_coords = read_court_coordinates(court_path)

    # Set up pose detector
    pose_detector = PoseDetect()

    # Define ROI margin
    roi_margin = 50

    roi_coords = []
    for coord in court_coords:
        x, y = coord
        x1, y1 = int(x - roi_margin), int(y - roi_margin)
        x2, y2 = int(x + roi_margin), int(y + roi_margin)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        roi_coords.append((x1, y1))
        roi_coords.append((x2, y2))

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'Class', 'Confidence', 'X', 'Y', 'Width', 'Height'])

        frame_number = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_number in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Check if it's a rally frame
            if rally_frames.get(frame_number, 0) == 1:
                # Crop frame to ROI
                roi_frame = frame[roi_coords[0][1]:roi_coords[1][1], roi_coords[0][0]:roi_coords[1][0]]

                # Run pose detection
                outputs, filtered_outputs = pose_detector.get_human_joints(roi_frame)

                # Process and write results to CSV
                for i in range(len(filtered_outputs)):
                    keypoints = np.array(filtered_outputs[i])  # 17, 2
                    keypoints = keypoints[:, :].reshape(-1, 2)
                    for p in range(keypoints.shape[0]):
                        x, y = keypoints[p, :]
                        x += roi_coords[0][0]
                        y += roi_coords[0][1]
                        csvwriter.writerow([frame_number, 'person', 1.0, x, y, 0, 0])

                    # Draw pose keypoints on the frame
                    roi_frame = pose_detector.draw_key_points(filtered_outputs, roi_frame)

                # Draw court lines
                court_poly = np.array(court_coords, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [court_poly], True, (255, 0, 0), 2)
                cv2.rectangle(frame, roi_coords[0], roi_coords[1], (0, 255, 0), 2)

                # Paste ROI frame back into original frame
                frame[roi_coords[0][1]:roi_coords[1][1], roi_coords[0][0]:roi_coords[1][0]] = roi_frame

            # Write the frame to the output video
            out.write(frame)

            # Debug message
            if frame_number % 100 == 0:
                print(f"Processed frame {frame_number} / {total_frames}")

    cap.release()
    out.release()
    print(f"Detection results saved to {csv_path}")
    print(f"Annotated video saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 pose.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    process_video(video_path)