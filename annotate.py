import os
import cv2
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

class VideoAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.base_name = os.path.splitext(os.path.basename(video_path))[0]
        self.shuttle_data = self.load_shuttle_data()
        self.preprocessed_data = self.load_preprocessed_data()
        self.court_points = self.load_court_points()
        self.pose_data = self.load_pose_data()
        self.video_capture = self.create_video_capture()
        self.video_writer = self.create_video_writer()

    def load_shuttle_data(self):
        csv_path = f"result/{self.base_name}_shuttle.csv"
        return pd.read_csv(csv_path)

    def load_preprocessed_data(self):
        return pd.read_csv(f"result/{self.base_name}_preprocessed.csv")

    def load_court_points(self):
        with open(f"result/{self.base_name}_court.txt", 'r') as file:
            return [tuple(map(float, line.strip().split(';'))) for line in file.readlines()]

    def load_pose_data(self):
        return pd.read_csv(f"result/{self.base_name}_pose.csv")

    def create_video_capture(self):
        return cv2.VideoCapture(self.video_path)

    def create_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        return cv2.VideoWriter(f"result/{self.base_name}_annotated.mp4", fourcc, fps, (width, height))

    def draw_court_lines(self, frame):
        for i in range(4):
            pt1 = (int(self.court_points[i][0]), int(self.court_points[i][1]))
            pt2 = (int(self.court_points[(i+1)%4][0]), int(self.court_points[(i+1)%4][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    def draw_shuttle(self, frame, frame_num):
        shuttle_row = self.shuttle_data[self.shuttle_data['Frame'] == frame_num]
        if not shuttle_row.empty:
            x, y = int(shuttle_row['X'].values[0]), int(shuttle_row['Y'].values[0])
            if x != 0 or y != 0:  # Only draw if position is not (0, 0)
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)  # Red dot for shuttle

    def draw_pose(self, frame, frame_num):
        frame_poses = self.pose_data[self.pose_data['frame_index'] == frame_num]

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
                print(f"No joints for human index {human_index} in frame {frame_num}")
                continue

            for joint_idx in range(len(human_keypoints)):
                if confidences[joint_idx] > 0.5:
                    x, y = human_keypoints[joint_idx]
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

    def annotate_frame(self, frame, frame_num):
        if frame_num < len(self.preprocessed_data) and self.preprocessed_data.iloc[frame_num]["is_rally_scene"]:
            self.draw_court_lines(frame)
        self.draw_shuttle(frame, frame_num)
        self.draw_pose(frame, frame_num)

    def run(self):
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Annotating video") as pbar:
            while self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                frame_num = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Adjust for 0-based indexing
                self.annotate_frame(frame, frame_num)
                self.video_writer.write(frame)
                pbar.update(1)

        self.video_capture.release()
        self.video_writer.release()

def main():
    parser = argparse.ArgumentParser(description="Annotate video with court lines, shuttle positions, and pose data")
    parser.add_argument("video_file", help="Path to the input video file")
    args = parser.parse_args()

    annotator = VideoAnnotator(args.video_file)
    annotator.run()
    print("Video annotation completed.")

if __name__ == "__main__":
    main()