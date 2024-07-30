import os
import cv2
import pandas as pd
import numpy as np
import argparse
import subprocess
import logging
from tqdm import tqdm

class VideoAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.base_name = os.path.splitext(os.path.basename(video_path))[0]
        self.result_dir = "result"
        os.makedirs(self.result_dir, exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def run_preprocess(self):
        logging.info("Running preprocessing step")
        subprocess.run(['python3', 'preprocess.py', self.video_path], check=True)
        logging.info("Preprocessing completed")

    def run_shuttle_detection(self):
        logging.info("Running shuttle detection")
        subprocess.run(['python3', 'shuttle.py', self.video_path], check=True)
        logging.info("Shuttle detection completed")

    def run_pose_detection(self):
        logging.info("Running pose detection")
        subprocess.run(['python3', 'pose.py', self.video_path], check=True)
        logging.info("Pose detection completed")

    def load_shuttle_data(self):
        csv_path = os.path.join(self.result_dir, f"{self.base_name}_shuttle.csv")
        logging.info(f"Loading shuttle data from {csv_path}")
        return pd.read_csv(csv_path)

    def load_preprocessed_data(self):
        csv_path = os.path.join(self.result_dir, f"{self.base_name}_preprocessed.csv")
        logging.info(f"Loading preprocessed data from {csv_path}")
        return pd.read_csv(csv_path)

    def load_court_points(self):
        court_file = os.path.join(self.result_dir, f"{self.base_name}_court.txt")
        logging.info(f"Loading court points from {court_file}")
        with open(court_file, 'r') as file:
            return [tuple(map(float, line.strip().split(';'))) for line in file.readlines()]

    def load_pose_data(self):
        csv_path = os.path.join(self.result_dir, f"{self.base_name}_pose.csv")
        logging.info(f"Loading pose data from {csv_path}")
        return pd.read_csv(csv_path)

    def create_video_capture(self):
        logging.info(f"Creating video capture for {self.video_path}")
        return cv2.VideoCapture(self.video_path)

    def create_video_writer(self, video_capture):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        output_path = os.path.join(self.result_dir, f"{self.base_name}_annotated.mp4")
        logging.info(f"Creating video writer for {output_path}")
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def draw_court_lines(self, frame, court_points):
        for i in range(4):
            pt1 = (int(court_points[i][0]), int(court_points[i][1]))
            pt2 = (int(court_points[(i+1)%4][0]), int(court_points[(i+1)%4][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    def draw_shuttle(self, frame, frame_num, shuttle_data):
        shuttle_row = shuttle_data[shuttle_data['Frame'] == frame_num]
        if not shuttle_row.empty:
            x, y = int(shuttle_row['X'].values[0]), int(shuttle_row['Y'].values[0])
            if x != 0 or y != 0:  # Only draw if position is not (0, 0)
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)  # Red dot for shuttle

    def draw_pose(self, frame, frame_num, pose_data):
        frame_poses = pose_data[pose_data['frame_index'] == frame_num]

        for human_index in frame_poses['human_index'].unique():
            human_keypoints = []
            confidences = []

            for _, row in frame_poses[frame_poses['human_index'] == human_index].iterrows():
                x, y = int(row['x']), int(row['y'])
                conf = row['confidence']
                if conf > 0.5 and x != 0 and y != 0:
                    human_keypoints.append((x, y))
                    confidences.append(conf)

            if len(human_keypoints) == 0:
                logging.debug(f"No joints for human index {human_index} in frame {frame_num}")
                continue

            for joint_idx in range(len(human_keypoints)):
                if confidences[joint_idx] > 0.5:
                    x, y = human_keypoints[joint_idx]
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

    def annotate_video(self):
        logging.info("Starting video annotation")
        shuttle_data = self.load_shuttle_data()
        preprocessed_data = self.load_preprocessed_data()
        court_points = self.load_court_points()
        pose_data = self.load_pose_data()

        video_capture = self.create_video_capture()
        video_writer = self.create_video_writer(video_capture)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Annotating video") as pbar:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break
                frame_num = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Adjust for 0-based indexing

                if frame_num < len(preprocessed_data) and preprocessed_data.iloc[frame_num]["is_rally_scene"]:
                    self.draw_court_lines(frame, court_points)
                self.draw_shuttle(frame, frame_num, shuttle_data)
                self.draw_pose(frame, frame_num, pose_data)

                video_writer.write(frame)
                pbar.update(1)

        video_capture.release()
        video_writer.release()
        logging.info("Video annotation completed")

    def run(self):
        self.run_preprocess()
        self.run_shuttle_detection()
        self.run_pose_detection()
        self.annotate_video()

def main():
    parser = argparse.ArgumentParser(description="Process and annotate badminton video")
    parser.add_argument("video_file", help="Path to the input video file")
    args = parser.parse_args()

    annotator = VideoAnnotator(args.video_file)
    annotator.run()

if __name__ == "__main__":
    main()