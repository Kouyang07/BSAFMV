import cv2
import numpy as np
import pandas as pd
import sys
import os

def read_court_points(court_csv):
    try:
        df = pd.read_csv(court_csv)
        df = df[~df['Point'].str.contains('NetPole')]
        return df[['Point', 'X', 'Y']].values
    except FileNotFoundError:
        print(f"Error: File {court_csv} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File {court_csv} is empty.")
        sys.exit(1)

def get_3d_court_points():
    # Define 3D court points (in meters)
    court_3d = np.array([
        [0, 0, 0], [0, 13.4, 0], [6.1, 13.4, 0], [6.1, 0, 0],  # P1, P2, P3, P4
        [0.46, 0, 0], [0.46, 13.4, 0], [5.64, 13.4, 0], [5.64, 0, 0],  # P5, P6, P7, P8
        [0, 4.715, 0], [6.1, 4.715, 0], [0, 8.68, 0], [6.1, 8.68, 0],  # P9, P10, P11, P12
        [3.05, 4.715, 0], [3.05, 8.68, 0], [0, 6.695, 0], [6.1, 6.695, 0],  # P13, P14, P15, P16
        [0, 0.76, 0], [6.1, 0.76, 0], [0, 12.64, 0], [6.1, 12.64, 0],  # P17, P18, P19, P20
        [3.05, 0, 0], [3.05, 13.4, 0]  # P21, P22
    ], dtype=np.float32)
    return court_3d

def calibrate_camera(court_2d, court_3d):
    # Estimate camera parameters
    camera_matrix = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float32)  # Initial guess
    dist_coeffs = np.zeros((4,1))  # Assume no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(court_3d, court_2d, camera_matrix, dist_coeffs)

    if not success:
        print("Failed to calibrate camera.")
        sys.exit(1)

    # Refine camera matrix
    optimal_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (1920, 1080), 1, (1920, 1080))

    return optimal_camera_matrix, dist_coeffs, rotation_vector, translation_vector

def project_3d_to_2d(points_3d, camera_matrix, dist_coeffs, rvec, tvec):
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

def transform_coordinates(input_csv, output_csv, court_csv):
    court_points = read_court_points(court_csv)
    court_2d = court_points[:, 1:3].astype(np.float32)
    court_3d = get_3d_court_points()

    camera_matrix, dist_coeffs, rvec, tvec = calibrate_camera(court_2d, court_3d)

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: File {input_csv} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File {input_csv} is empty.")
        sys.exit(1)

    # Prepare 3D points for reprojection
    points_3d = np.column_stack((df[['X', 'Y']].values, np.zeros(len(df))))

    # Project 3D points to 2D
    points_2d = project_3d_to_2d(points_3d, camera_matrix, dist_coeffs, rvec, tvec)

    df[['X', 'Y']] = points_2d

    # Filter out unrealistic points
    court_width, court_length = 6.1, 13.4
    df = df[(df['X'] >= 0) & (df['X'] <= court_width) &
            (df['Y'] >= 0) & (df['Y'] <= court_length)]

    df.to_csv(output_csv, index=False)

    # Calculate reprojection error
    court_2d_reprojected = project_3d_to_2d(court_3d, camera_matrix, dist_coeffs, rvec, tvec)
    reprojection_error = np.mean(np.linalg.norm(court_2d - court_2d_reprojected, axis=1))

    print(f"Reprojection error: {reprojection_error:.4f} pixels")

    return df, reprojection_error, camera_matrix, dist_coeffs, rvec, tvec

def draw_court(court, left, right, top, bottom):
    cv2.line(court, (left, bottom), (left, top), (255, 255, 255), 2)  # Left line
    cv2.line(court, (right, bottom), (right, top), (255, 255, 255), 2)  # Right line
    cv2.line(court, (left, bottom), (right, bottom), (255, 255, 255), 2)  # Bottom line
    cv2.line(court, (left, top), (right, top), (255, 255, 255), 2)  # Top line
    cv2.line(court, (left, (top + bottom) // 2), (right, (top + bottom) // 2), (255, 255, 255), 1)  # Net

def create_video(df, output_video, reprojection_error, camera_params):
    court_width, court_length = 6.1, 13.4
    padding = 1  # 1 meter padding on each side
    scale = 100  # 100 pixels per meter

    width_px = int((court_width + 2*padding) * scale)
    height_px = int((court_length + 2*padding) * scale)

    court = np.zeros((height_px, width_px, 3), np.uint8)

    left = int(padding * scale)
    right = int((padding + court_width) * scale)
    top = int(padding * scale)
    bottom = int((padding + court_length) * scale)

    draw_court(court, left, right, top, bottom)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter(output_video, fourcc, fps, (width_px, height_px))

    camera_matrix, dist_coeffs, rvec, tvec = camera_params

    for index, row in df.iterrows():
        court_copy = court.copy()

        frame_info = f"Frame: {int(row['Frame'])}"
        position_info = f"X: {row['X']:.2f}, Y: {row['Y']:.2f}"
        accuracy_info = f"Reprojection Error: {reprojection_error:.4f} pixels"
        camera_info = f"Camera: f={camera_matrix[0,0]:.1f}, cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}"

        cv2.putText(court_copy, frame_info, (10, height_px - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(court_copy, position_info, (10, height_px - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(court_copy, accuracy_info, (10, height_px - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(court_copy, camera_info, (10, height_px - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        x = int((row['X'] + padding) * scale)
        y = int((row['Y'] + padding) * scale)
        cv2.circle(court_copy, (x, y), 5, (0, 255, 0), -1)
        out.write(court_copy)

    out.release()

def main(input_video):
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    input_csv = f'result/{base_name}_shuttle.csv'
    court_csv = f'result/{base_name}_court.csv'
    transformed_csv = f'result/{base_name}_shuttle_transformed.csv'
    output_video = f'result/{base_name}_visualization.mp4'

    df, reprojection_error, camera_matrix, dist_coeffs, rvec, tvec = transform_coordinates(input_csv, transformed_csv, court_csv)
    camera_params = (camera_matrix, dist_coeffs, rvec, tvec)
    create_video(df, output_video, reprojection_error, camera_params)

    print(f"Visualization video created: {output_video}")
    print(f"Camera Matrix:\n{camera_matrix}")
    print(f"Distortion Coefficients: {dist_coeffs.ravel()}")
    print(f"Rotation Vector: {rvec.ravel()}")
    print(f"Translation Vector: {tvec.ravel()}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 video.py <input_video_path>")
        sys.exit(1)

    input_video = sys.argv[1]
    main(input_video)