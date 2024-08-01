import cv2
import torch
import numpy as np
import csv
import sys
import os
import logging
from tqdm import tqdm
from ultralytics import YOLO


def load_court_coordinates(court_file):
    logging.info(f"Reading court coordinates from {court_file}")
    coordinates = {}
    with open(court_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            coordinates[row['Point']] = (float(row['X']), float(row['Y']))
    logging.info(f"Court coordinates: {coordinates}")
    return coordinates


class PoseDetect:
    def __init__(self):
        self.device = self.select_device()
        self.setup_YOLO()

    def reset(self):
        self.got_info = False

    def select_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            logging.warning("MPS device detected, but falling back to CPU for unsupported operations.")
            return "cpu"
        else:
            return "cpu"

    def setup_YOLO(self):
        logging.info(f"Loading model on device: {self.device}")
        self.__pose_model = YOLO('yolov8x-pose.pt', verbose=False)  # disable debug info
        self.__pose_model.to(self.device)

    def del_YOLO(self):
        logging.info("Deleting model")
        del self.__pose_model

    def get_human_joints(self, frame):
        results = self.__pose_model(frame)
        return results

    def draw_key_points(self, results, image, human_limit=-1, conf_threshold=0.3):
        image_copy = image.copy()
        if results[0].keypoints is None:
            return image_copy  # Return original image if no keypoints detected

        keypoints = results[0].keypoints

        edges = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
                 (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

        top_color_edge = (255, 0, 0)
        bot_color_edge = (0, 0, 255)
        top_color_joint = (115, 47, 14)
        bot_color_joint = (35, 47, 204)

        for i, person_keypoints in enumerate(keypoints):
            if i > human_limit and human_limit != -1:
                break

            color = top_color_edge if i == 0 else bot_color_edge
            color_joint = top_color_joint if i == 0 else bot_color_joint

            if person_keypoints.xy is None or person_keypoints.conf is None:
                continue

            xy = person_keypoints.xy[0]
            conf = person_keypoints.conf[0]

            for p in range(xy.shape[0]):
                if conf[p].item() > conf_threshold:
                    x, y = int(xy[p, 0].item()), int(xy[p, 1].item())
                    cv2.circle(image_copy, (x, y), 3, color_joint, thickness=-1, lineType=cv2.FILLED)

            for e in edges:
                if e[0] < xy.shape[0] and e[1] < xy.shape[0]:
                    if conf[e[0]].item() > conf_threshold and conf[e[1]].item() > conf_threshold:
                        start_point = (int(xy[e[0], 0].item()), int(xy[e[0], 1].item()))
                        end_point = (int(xy[e[1], 0].item()), int(xy[e[1], 1].item()))
                        cv2.line(image_copy, start_point, end_point, color, 2, lineType=cv2.LINE_AA)

        return image_copy

    def is_point_in_court(self, point, court_points):
        x, y = point
        court_polygon = [
            court_points['P1'], court_points['P2'], court_points['P3'], court_points['P4']
        ]
        n = len(court_polygon)
        inside = False
        p1x, p1y = court_polygon[0]
        for i in range(n+1):
            p2x, p2y = court_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y-p1y) * (p2x-p1x) / (p2y-p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def process_video(self, input_video_path, output_video_path, output_csv_path, preprocessed_csv_path):
        logging.info(f"Processing video: {input_video_path}")
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        csv_data = []

        # Read preprocessed CSV
        preprocessed_frames = set()
        with open(preprocessed_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['is_rally_scene'] == 'True':
                    preprocessed_frames.add(int(row['frame']))

        # Load court coordinates
        court_points_file = f"result/{os.path.splitext(os.path.basename(input_video_path))[0]}_court.csv"
        court_points = load_court_coordinates(court_points_file)

        with tqdm(total=frame_count, desc="Processing video frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index in preprocessed_frames:
                    results = self.get_human_joints(frame)
                    frame_with_keypoints = frame.copy()

                    if results[0].keypoints is not None:
                        keypoints = results[0].keypoints
                        for human_idx, person_keypoints in enumerate(keypoints):
                            if person_keypoints.xy is None or person_keypoints.conf is None:
                                continue
                            xy = person_keypoints.xy[0]
                            conf = person_keypoints.conf[0]

                            # Check if any body part is within the court
                            is_in_court = False
                            for joint_idx in range(xy.shape[0]):
                                if conf[joint_idx].item() > 0.5:
                                    x, y = int(xy[joint_idx, 0].item()), int(xy[joint_idx, 1].item())
                                    if self.is_point_in_court((x, y), court_points):
                                        is_in_court = True
                                        print(f"Joint {joint_idx} is in court at frame {frame_index}")
                                        break

                            if is_in_court:
                                # Draw keypoints for this person
                                for joint_idx in range(xy.shape[0]):
                                    if conf[joint_idx].item() > 0.5:
                                        x, y = int(xy[joint_idx, 0].item()), int(xy[joint_idx, 1].item())
                                        cv2.circle(frame_with_keypoints, (x, y), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

                                # Draw edges for this person
                                edges = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
                                         (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
                                for e in edges:
                                    if e[0] < xy.shape[0] and e[1] < xy.shape[0]:
                                        if conf[e[0]].item() > 0.5 and conf[e[1]].item() > 0.5:
                                            start_point = (int(xy[e[0], 0].item()), int(xy[e[0], 1].item()))
                                            end_point = (int(xy[e[1], 0].item()), int(xy[e[1], 1].item()))
                                            cv2.line(frame_with_keypoints, start_point, end_point, (0, 255, 0), 2, lineType=cv2.LINE_AA)

                                # Append data to CSV
                                for joint_idx in range(xy.shape[0]):
                                    csv_data.append([frame_index, human_idx, joint_idx,
                                                     xy[joint_idx, 0].item(),
                                                     xy[joint_idx, 1].item(),
                                                     conf[joint_idx].item()])
                            else:
                                print(f"No joints in court at frame {frame_index}")
                    else:
                        frame_with_keypoints = frame
                        print(f"No keypoints detected at frame {frame_index}")
                else:
                    frame_with_keypoints = frame

                out.write(frame_with_keypoints)
                frame_index += 1
                pbar.update(1)

        cap.release()
        out.release()

        with open(output_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['frame_index', 'human_index', 'joint_index', 'x', 'y', 'confidence'])
            csvwriter.writerows(csv_data)

        logging.info(f"Processed video saved as: {output_video_path}")
        logging.info(f"Data saved as: {output_csv_path}")

def main(input_video_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = f"result/{base_name}_pose.mp4"
    output_csv_path = f"result/{base_name}_pose.csv"
    preprocessed_csv_path = f"result/{base_name}_preprocessed.csv"

    pose_detect = PoseDetect()
    pose_detect.process_video(input_video_path, output_video_path, output_csv_path, preprocessed_csv_path)
    logging.info(f"Processed video saved as: {output_video_path}")
    logging.info(f"Data saved as: {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 pose.py <input_video_path>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    main(input_video_path)