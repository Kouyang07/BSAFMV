import cv2
import torch
import numpy as np
import csv
import sys
import os
import logging
from tqdm import tqdm
from ultralytics import YOLO

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

    def draw_key_points(self, results, image, human_limit=-1):
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

            xy = person_keypoints.xy[0]
            for p in range(xy.shape[0]):
                x, y = int(xy[p, 0].item()), int(xy[p, 1].item())
                cv2.circle(image_copy, (x, y), 3, color_joint, thickness=-1, lineType=cv2.FILLED)

            for e in edges:
                if e[0] < xy.shape[0] and e[1] < xy.shape[0]:
                    start_point = (int(xy[e[0], 0].item()), int(xy[e[0], 1].item()))
                    end_point = (int(xy[e[1], 0].item()), int(xy[e[1], 1].item()))
                    cv2.line(image_copy, start_point, end_point, color, 2, lineType=cv2.LINE_AA)

        return image_copy

    def process_video(self, input_video_path, output_video_path, output_csv_path):
        logging.info(f"Processing video: {input_video_path}")
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        csv_data = []

        with tqdm(total=frame_count, desc="Processing video frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.get_human_joints(frame)
                frame_with_keypoints = self.draw_key_points(results, frame)
                out.write(frame_with_keypoints)

                if results[0].keypoints is not None:
                    keypoints = results[0].keypoints
                    for human_idx, person_keypoints in enumerate(keypoints):
                        xy = person_keypoints.xy[0]
                        for joint_idx in range(xy.shape[0]):
                            csv_data.append([frame_index, human_idx, joint_idx,
                                             xy[joint_idx, 0].item(),
                                             xy[joint_idx, 1].item()])

                frame_index += 1
                pbar.update(1)

        cap.release()
        out.release()

        with open(output_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['frame_index', 'human_index', 'joint_index', 'x', 'y'])
            csvwriter.writerows(csv_data)

        logging.info(f"Processed video saved as: {output_video_path}")
        logging.info(f"Data saved as: {output_csv_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if len(sys.argv) != 2:
        print("Usage: python3 pose.py {path to video}")
        sys.exit(1)

    input_video_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = f"result/{base_name}_pose.mp4"
    output_csv_path = f"result/{base_name}_pose.csv"

    pose_detect = PoseDetect()
    pose_detect.process_video(input_video_path, output_video_path, output_csv_path)
    print(f"Processed video saved as: {output_video_path}")
    print(f"Data saved as: {output_csv_path}")