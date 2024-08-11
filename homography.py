import cv2
import numpy as np
import pandas as pd
import sys
import os

def read_court_points(court_csv):
    return pd.read_csv(court_csv).query('~Point.str.contains("NetPole")')[['X', 'Y']].values

def calculate_reprojection_error(pts_src, pts_dst, h):
    pts_reprojected = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2).astype(np.float32), h)
    return np.sqrt(np.mean(np.sum((pts_dst.reshape(-1, 1, 2) - pts_reprojected) ** 2, axis=2)))

def transform_coordinates(input_csv, output_csv, court_csv):
    pts_src = read_court_points(court_csv)
    pts_dst = np.array([[0, 13.4], [0, 0], [6.1, 0], [6.1, 13.4], [0.46, 13.4], [0.46, 0],
                        [5.64, 0], [5.64, 13.4], [0, 4.715], [6.1, 4.715], [0, 8.68], [6.1, 8.68],
                        [3.05, 4.715], [3.05, 8.68], [0, 6.695], [6.1, 6.695], [0, 0.76],
                        [6.1, 0.76], [0, 12.64], [6.1, 12.64], [3.05, 0], [3.05, 13.4]], dtype=np.float32)

    h, _ = cv2.findHomography(pts_src, pts_dst)
    reprojection_error = calculate_reprojection_error(pts_src, pts_dst, h)
    print(f"Reprojection error: {reprojection_error:.4f} meters")

    df = pd.read_csv(input_csv)
    df[['X', 'Y']] = cv2.perspectiveTransform(df[['X', 'Y']].values.reshape(-1, 1, 2).astype(np.float32), h).reshape(-1, 2)
    df.to_csv(output_csv, index=False)
    return df, reprojection_error

def create_visualization_video(df, original_video_path, output_video, reprojection_error):
    original_cap = cv2.VideoCapture(original_video_path)
    original_width, original_height = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(original_cap.get(cv2.CAP_PROP_FPS))

    scale = original_height / 15.4
    width_px, height_px = int(8.1 * scale), original_height

    court = np.zeros((height_px, width_px, 3), np.uint8)
    left, right, top, bottom = int(scale), int(7.1 * scale), int(scale), int(14.4 * scale)

    cv2.rectangle(court, (left, top), (right, bottom), (255, 255, 255), 2)
    cv2.line(court, (left, (top + bottom) // 2), (right, (top + bottom) // 2), (255, 255, 255), 1)

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (original_width + width_px, height_px))

    for index, row in df.iterrows():
        ret, original_frame = original_cap.read()
        if not ret: break

        court_copy = court.copy()
        x, y = int((row['X'] + 1) * scale), int((13.4 - row['Y'] + 1) * scale)
        cv2.circle(court_copy, (x, y), 5, (0, 255, 0), -1)

        cv2.putText(court_copy, f"Frame: {int(row['Frame'])}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(court_copy, f"X: {row['X']:.2f}, Y: {row['Y']:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(court_copy, f"Homography Error: {reprojection_error:.4f} m", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        out.write(np.hstack((original_frame, court_copy)))

    original_cap.release()
    out.release()

def main(input_video):
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    input_csv, court_csv = f'result/{base_name}_shuttle.csv', f'result/{base_name}_court.csv'
    transformed_csv, output_video = f'result/{base_name}_shuttle_transformed.csv', f'result/{base_name}_visualization.mp4'

    df, reprojection_error = transform_coordinates(input_csv, transformed_csv, court_csv)
    create_visualization_video(df, input_video, output_video, reprojection_error)
    print(f"Visualization video created: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 video.py <input_video_path>")
        sys.exit(1)

    main(sys.argv[1])