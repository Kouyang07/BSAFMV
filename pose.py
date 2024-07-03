import cv2
import torch
import torchvision
import numpy as np
import copy
import csv
import sys
import os
import logging
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import functional as F

class PoseDetect:
    def __init__(self):
        self.device = self.select_device()
        self.setup_RCNN()

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

    def setup_RCNN(self):
        logging.info(f"Loading model on device: {self.device}")
        self.__pose_kpRCNN = torch.load('resources/pose_kpRCNN.pth', map_location=torch.device(self.device))
        self.__pose_kpRCNN.to(self.device).eval()

    def del_RCNN(self):
        logging.info("Deleting model")
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

    def process_video(self, input_video_path, output_video_path, output_csv_path):
        logging.info(f"Processing video: {input_video_path}")
        cap = cv2.VideoCapture(input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        csv_data = []

        with tqdm(total=frame_count, desc="Processing video frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                outputs, filtered_outputs = self.get_human_joints(frame)
                frame_with_keypoints = self.draw_key_points(filtered_outputs, frame)
                out.write(frame_with_keypoints)

                for human_idx, joints in enumerate(filtered_outputs):
                    for joint_idx, joint in enumerate(joints):
                        csv_data.append([frame_index, human_idx, joint_idx, joint[0], joint[1]])

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