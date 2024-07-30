import cv2
import numpy as np
import pandas as pd
import sys
import os

def transform_coordinates(input_csv, output_csv):
    # Define the source and destination points
    pts_src = np.array([[273.146, 386.079], [135.366, 703.384], [966.518, 701.338], [799.461, 387.37]])
    pts_dst = np.array([[0, 13.4], [0, 0], [6.1, 0], [6.1, 13.4]])

    # Find the homography matrix
    h, _ = cv2.findHomography(pts_src, pts_dst)

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
    return df

def create_video(df, output_video):
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
    cv2.line(court, (left, top), (left, bottom), (255, 255, 255), 2)  # Left line
    cv2.line(court, (right, top), (right, bottom), (255, 255, 255), 2)  # Right line
    cv2.line(court, (left, top), (right, top), (255, 255, 255), 2)  # Top line
    cv2.line(court, (left, bottom), (right, bottom), (255, 255, 255), 2)  # Bottom line
    cv2.line(court, (left, (top + bottom) // 2), (right, (top + bottom) // 2), (255, 255, 255), 1)  # Net

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    out = cv2.VideoWriter(output_video, fourcc, fps, (width_px, height_px))

    # Draw the shuttle positions on the court
    for index, row in df.iterrows():
        court_copy = court.copy()
        cv2.putText(court_copy, f"Frame: {int(row['Frame'])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        x = int((row['X'] + padding) * scale)
        y = int((court_length - row['Y'] + padding) * scale)  # Flip Y-axis
        cv2.circle(court_copy, (x, y), 5, (0, 255, 0), -1)
        out.write(court_copy)

    # Release the video writer
    out.release()

def main(input_video):
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    input_csv = f'result/{base_name}_shuttle.csv'
    transformed_csv = f'result/{base_name}_shuttle_transformed.csv'
    output_video = f'result/{base_name}_visualization.mp4'

    # Transform coordinates
    df = transform_coordinates(input_csv, transformed_csv)

    # Create visualization video
    create_video(df, output_video)

    print(f"Visualization video created: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 video.py <input_video_path>")
        sys.exit(1)

    input_video = sys.argv[1]
    main(input_video)