import cv2
import numpy as np
import pandas as pd
import sys
import os

def read_court_points(court_csv):
    df = pd.read_csv(court_csv)
    # Exclude net pole points
    df = df[~df['Point'].str.contains('NetPole')]
    return df[['X', 'Y']].values

def calculate_reprojection_error(pts_src, pts_dst, h):
    pts_src = pts_src.reshape(-1, 1, 2)
    pts_dst = pts_dst.reshape(-1, 1, 2)
    pts_reprojected = cv2.perspectiveTransform(pts_src.astype(np.float32), h)
    error = np.sqrt(np.mean(np.sum((pts_dst - pts_reprojected) ** 2, axis=2)))
    return error

def transform_coordinates(input_csv, output_csv, court_csv):
    # Read the court points
    pts_src = read_court_points(court_csv)

    # Define the destination points (in meters)
    pts_dst = np.array([
        [0, 13.4], [0, 0], [6.1, 0], [6.1, 13.4],  # P1, P2, P3, P4
        [0.46, 13.4], [0.46, 0], [5.64, 0], [5.64, 13.4],  # P5, P6, P7, P8
        [0, 4.715], [6.1, 4.715], [0, 8.68], [6.1, 8.68],  # P9, P10, P11, P12
        [3.05, 4.715], [3.05, 8.68], [0, 6.695], [6.1, 6.695],  # P13, P14, P15, P16
        [0, 0.76], [6.1, 0.76], [0, 12.64], [6.1, 12.64],  # P17, P18, P19, P20
        [3.05, 0], [3.05, 13.4]  # P21, P22
    ])

    # Find the homography matrix
    h, _ = cv2.findHomography(pts_src, pts_dst)

    # Calculate reprojection error
    reprojection_error = calculate_reprojection_error(pts_src, pts_dst, h)
    print(f"Reprojection error: {reprojection_error:.4f} meters")

    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Apply the transformation to each frame
    for index, row in df.iterrows():
        point_src = np.array([[row['X'], row['Y']]], dtype=np.float32)
        point_src = point_src.reshape(-1, 1, 2)
        point_dst = cv2.perspectiveTransform(point_src, h)
        x = float(point_dst[0][0][0])
        y = float(point_dst[0][0][1])
        df.at[index, 'X'] = x
        df.at[index, 'Y'] = y

    df.to_csv(output_csv, index=False)
    return df, reprojection_error

def create_video(df, output_video, reprojection_error):
    # Define court dimensions in meters
    court_width = 6.1
    court_length = 13.4

    # Define padding (in meters)
    padding = 1  # 1 meter padding on each side

    # Define scaling factor (pixels per meter)
    scale = 100  # 100 pixels per meter

    # Calculate dimensions in pixels
    width_px = int((court_width + 2*padding) * scale)
    height_px = int((court_length + 2*padding) * scale)

    # Create a black background image
    court = np.zeros((height_px, width_px, 3), np.uint8)

    # Calculate court coordinates in pixels
    left = int(padding * scale)
    right = int((padding + court_width) * scale)
    top = int(padding * scale)
    bottom = int((padding + court_length) * scale)

    # Draw the badminton court
    cv2.line(court, (left, bottom), (left, top), (255, 255, 255), 2)  # Left line
    cv2.line(court, (right, bottom), (right, top), (255, 255, 255), 2)  # Right line
    cv2.line(court, (left, bottom), (right, bottom), (255, 255, 255), 2)  # Bottom line
    cv2.line(court, (left, top), (right, top), (255, 255, 255), 2)  # Top line
    cv2.line(court, (left, (top + bottom) // 2), (right, (top + bottom) // 2), (255, 255, 255), 1)  # Net

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    out = cv2.VideoWriter(output_video, fourcc, fps, (width_px, height_px))

    # Draw the shuttle positions on the court
    for index, row in df.iterrows():
        court_copy = court.copy()

        # Display frame number, x, y position, and homography accuracy
        frame_info = f"Frame: {int(row['Frame'])}"
        position_info = f"X: {row['X']:.2f}, Y: {row['Y']:.2f}"
        accuracy_info = f"Homography Error: {reprojection_error:.4f} m"

        cv2.putText(court_copy, frame_info, (10, height_px // 2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(court_copy, position_info, (10, height_px // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(court_copy, accuracy_info, (10, height_px // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        x = int((row['X'] + padding) * scale)
        y = int((court_length - row['Y'] + padding) * scale)  # Flip Y-axis
        cv2.circle(court_copy, (x, y), 5, (0, 255, 0), -1)
        out.write(court_copy)

    # Release the video writer
    out.release()

def main(input_video):
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    input_csv = f'result/{base_name}_shuttle.csv'
    court_csv = f'result/{base_name}_court.csv'
    transformed_csv = f'result/{base_name}_shuttle_transformed.csv'
    output_video = f'result/{base_name}_visualization.mp4'

    # Transform coordinates and get reprojection error
    df, reprojection_error = transform_coordinates(input_csv, transformed_csv, court_csv)

    # Create visualization video
    create_video(df, output_video, reprojection_error)

    print(f"Visualization video created: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 video.py <input_video_path>")
        sys.exit(1)

    input_video = sys.argv[1]
    main(input_video)