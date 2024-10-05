import sys
import os
import numpy as np
import cv2
import pandas as pd
import argparse
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VideoCaptureContext:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

    def __enter__(self):
        return self.cap

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()


class VideoWriterContext:
    def __init__(self, filename, fourcc, fps, frame_size):
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise IOError(f"Cannot open video writer with filename {filename}")

    def __enter__(self):
        return self.writer

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.release()


def define_world_points():
    """
    Defines the 3D world points of the badminton court (court landmarks).

    Returns:
        np.ndarray: An array of world points in meters (X, Y, Z).
    """
    # World coordinates in meters (X, Y, Z)
    world_points = np.array([
        [0, 0, 0],         # P1 (Bottom left corner)
        [0, 13.4, 0],      # P2 (Top left corner)
        [6.1, 13.4, 0],    # P3 (Top right corner)
        [6.1, 0, 0],       # P4 (Bottom right corner)
        [0.46, 0, 0],      # P5
        [0.46, 13.4, 0],   # P6
        [5.64, 13.4, 0],   # P7
        [5.64, 0, 0],      # P8
        [0, 4.715, 0],     # P9
        [6.1, 4.715, 0],   # P10
        [0, 8.68, 0],      # P11
        [6.1, 8.68, 0],    # P12
        [3.05, 4.715, 0],  # P13
        [3.05, 8.68, 0],   # P14
        [0, 6.695, 0],     # P15
        [6.1, 6.695, 0],   # P16
        [0, 0.76, 0],      # P17
        [6.1, 0.76, 0],    # P18
        [0, 12.64, 0],     # P19
        [6.1, 12.64, 0],   # P20
        [3.05, 0, 0],      # P21
        [3.05, 13.4, 0]    # P22
    ], dtype=np.float32)
    return world_points


def read_court_points(court_csv):
    """
    Reads the 2D court image points from the CSV file.

    Args:
        court_csv (str): Path to the court landmarks CSV file.

    Returns:
        np.ndarray: An array of image points in pixels (X, Y).

    Raises:
        FileNotFoundError: If the court CSV file is not found.
        pd.errors.EmptyDataError: If the court CSV file is empty.
    """
    try:
        df = pd.read_csv(court_csv)
        df = df[~df['Point'].str.contains('NetPole', na=False)]  # Exclude net poles if present
        image_points = df[['X', 'Y']].values.astype(np.float32)
        return image_points
    except FileNotFoundError as e:
        logging.error(f"File {court_csv} not found.")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"File {court_csv} is empty.")
        raise e


def calibrate_camera(world_points, image_points, image_size):
    """
    Performs camera calibration to obtain intrinsic and extrinsic parameters.

    Args:
        world_points (np.ndarray): Array of world points.
        image_points (np.ndarray): Array of corresponding image points.
        image_size (tuple): Size of the image (width, height).

    Returns:
        tuple: camera_matrix, dist_coeffs, rvec, tvec

    Raises:
        RuntimeError: If camera calibration fails.
    """
    # Prepare object points and image points
    obj_points = [world_points]
    img_points = [image_points]

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)

    if not ret:
        logging.error("Camera calibration failed.")
        raise RuntimeError("Camera calibration failed.")

    # Extract rotation and translation vectors
    rvec = rvecs[0]
    tvec = tvecs[0]

    logging.info("Camera Calibration Results:")
    logging.info("Camera Matrix:\n%s", camera_matrix)
    logging.info("Distortion Coefficients:\n%s", dist_coeffs)
    logging.info("Rotation Vector:\n%s", rvec)
    logging.info("Translation Vector:\n%s", tvec)

    return camera_matrix, dist_coeffs, rvec, tvec


def compute_homography(world_points, image_points):
    """
    Computes the homography from world coordinates to image coordinates.

    Args:
        world_points (np.ndarray): Array of world points.
        image_points (np.ndarray): Array of image points.

    Returns:
        np.ndarray: Homography matrix H.

    Raises:
        cv2.error: If homography computation fails.
    """
    # Since the court is on the plane z=0, we can compute a homography
    # between the 2D court coordinates and the image points
    world_points_2d = world_points[:, :2]
    H, status = cv2.findHomography(world_points_2d, image_points)
    if H is None:
        logging.error("Homography computation failed.")
        raise cv2.error("Homography computation failed.")
    return H


def read_pose_data(pose_csv):
    """
    Reads the pose estimation data from the CSV file.

    Args:
        pose_csv (str): Path to the pose estimation CSV file.

    Returns:
        pd.core.groupby.generic.DataFrameGroupBy: Grouped pose data by frame and human.

    Raises:
        FileNotFoundError: If the pose CSV file is not found.
        pd.errors.EmptyDataError: If the pose CSV file is empty.
    """
    try:
        df = pd.read_csv(pose_csv)
        # Group data by frame and human
        grouped = df.groupby(['frame_index', 'human_index'])
        return grouped
    except FileNotFoundError as e:
        logging.error(f"File {pose_csv} not found.")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"File {pose_csv} is empty.")
        raise e


def compute_player_locations(grouped):
    """
    Computes the player's foot positions as the midpoint between both ankles (left and right) in image coordinates.

    Args:
        grouped (pd.core.groupby.generic.DataFrameGroupBy): Grouped pose data by frame and human.

    Returns:
        dict: A dictionary with keys as (frame_index, human_index) and values as image coordinates.
    """
    results = {}
    for (frame_index, human_index), joints in grouped:
        # Use joints 15 and 16 for ankles
        ankle_joints = joints[joints['joint_index'].isin([15, 16])]
        high_confidence = ankle_joints[ankle_joints['confidence'] > 0.7]

        if len(high_confidence) == 2:
            # If both ankles have high confidence, compute the midpoint
            x = high_confidence['x'].mean()
            y = high_confidence['y'].mean()
        elif len(high_confidence) == 1:
            # If only one ankle has high confidence, use it
            x = high_confidence.iloc[0]['x']
            y = high_confidence.iloc[0]['y']
        else:
            # If no joints with high confidence, skip this player
            continue

        key = (frame_index, human_index)
        results[key] = np.array([x, y], dtype=np.float32)
    return results


def draw_court(image, court_height_px, court_width_px):
    """
    Draws the court lines on the schematic court image.

    Args:
        image (np.ndarray): The court image to draw on.
        court_height_px (int): The height of the court image in pixels.
        court_width_px (int): The width of the court image in pixels.
    """
    # Draw outer boundaries
    cv2.rectangle(image, (0, 0), (court_width_px - 1, court_height_px - 1), (255, 255, 255), 2)
    # Draw the net at half the length
    net_y = int(court_height_px / 2)
    cv2.line(image, (0, net_y), (court_width_px - 1, net_y), (255, 255, 255), 2)
    # Optionally, draw additional court lines (e.g., service lines)
    # For simplicity, we'll only draw boundaries and net here

def backproject_to_plane(image_point, camera_matrix, dist_coeffs, R, t, plane_z=0):
    """
    Backprojects an image point to the plane z=plane_z in world coordinates.

    Args:
        image_point (np.ndarray): The image point (u, v).
        camera_matrix (np.ndarray): The camera matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        R (np.ndarray): The rotation matrix.
        t (np.ndarray): The translation vector.
        plane_z (float): The z-coordinate of the plane in world coordinates.

    Returns:
        np.ndarray: The world point (X, Y, Z) on the plane z=plane_z.
    """
    # Undistort the image point
    undistorted_point = cv2.undistortPoints(
        np.array([[image_point]], dtype=np.float32), camera_matrix, dist_coeffs, P=None)

    # The undistorted point is in normalized image coordinates
    x = undistorted_point[0, 0, 0]
    y = undistorted_point[0, 0, 1]

    # Construct a vector in camera coordinates
    point_cam = np.array([x, y, 1.0])

    # The camera center in world coordinates
    origin_world = -R.T @ t

    # The direction vector in world coordinates
    direction_world = R.T @ point_cam

    # Now, compute the intersection with the plane z=plane_z
    s = (plane_z - origin_world[2]) / direction_world[2]
    p_world = origin_world + s * direction_world

    return p_world

def main():
    """
    Main function to process and create the visualization video.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process and visualize badminton court video with player positions.')
    parser.add_argument('video_path', type=str, help='Path to input video file.')
    parser.add_argument('--output_video', type=str, help='Path to output video file.')
    args = parser.parse_args()

    try:
        # Extract base name of the video file
        video_base_name = os.path.basename(args.video_path)
        video_name_without_ext = os.path.splitext(video_base_name)[0]

        # Construct paths to pose CSV and court CSV files
        pose_csv = os.path.join('result', f'{video_name_without_ext}_pose.csv')
        court_csv = os.path.join('result', f'{video_name_without_ext}_court.csv')

        # Set output video path
        if args.output_video:
            output_video_path = args.output_video
        else:
            output_video_path = f'result/{video_name_without_ext}_player_positions.mp4'

        # Step 1: Camera Calibration
        world_points = define_world_points()
        image_points = read_court_points(court_csv)

        # Open the original video to get the image size and FPS
        with VideoCaptureContext(args.video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            image_size = (original_width, original_height)  # Width, Height

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Perform camera calibration
            camera_matrix, dist_coeffs, rvec, tvec = calibrate_camera(world_points, image_points, image_size)

            # Compute rotation matrix and translation vector
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)

            # Step 2: Read Pose Data and Compute Player Locations
            grouped = read_pose_data(pose_csv)
            player_image_points = compute_player_locations(grouped)

            # Organize results by frame
            frames = {}
            for (frame_index, human_index), img_point in player_image_points.items():
                if frame_index not in frames:
                    frames[frame_index] = {}
                frames[frame_index][human_index] = img_point

            # Define the size of the court image
            court_height_px = 720  # Adjust this value as needed
            court_length_m = 13.4  # meters
            court_scale = court_height_px / court_length_m  # pixels per meter
            court_width_m = 6.1  # meters
            court_width_px = int(court_width_m * court_scale)

            # Create a blank court image template
            court_image_template = np.zeros((court_height_px, court_width_px, 3), dtype=np.uint8)

            # Draw the court lines on the template
            draw_court(court_image_template, court_height_px, court_width_px)

            # Adjust output video properties
            output_width = original_width + court_width_px
            output_height = max(original_height, court_height_px)

            # Create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

            with VideoWriterContext(output_video_path, fourcc, fps, (output_width, output_height)) as output_video:
                # Generate frames and write to video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
                for frame_index in tqdm(range(frame_count), desc='Processing frames'):
                    ret, original_frame = cap.read()
                    if not ret:
                        break

                    # Create a copy of the court image
                    court_image = court_image_template.copy()

                    # Get frame data
                    frame_data = frames.get(frame_index, {})

                    # Plot player positions
                    for human_index in sorted(frame_data.keys()):
                        img_point = frame_data[human_index]

                        # Backproject the image point to the court plane z=0
                        world_point = backproject_to_plane(img_point, camera_matrix, dist_coeffs, R, t, plane_z=0)
                        X_world, Y_world = world_point[0], world_point[1]

                        # Convert from court coordinates (meters) to pixel coordinates on the court image
                        pixel_x = int(X_world * court_scale)
                        pixel_y = int(Y_world * court_scale)  # Adjusted mapping

                        # Ensure pixel coordinates are within court image boundaries
                        if 0 <= pixel_x < court_width_px and 0 <= pixel_y < court_height_px:
                            # Plot the player's position on the court image
                            color = (0, 0, 255) if human_index == 0 else (0, 255, 0)  # Different colors for players
                            cv2.circle(court_image, (pixel_x, pixel_y), 5, color, -1)
                            cv2.putText(court_image, f'P{human_index + 1}', (pixel_x + 5, pixel_y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        else:
                            logging.warning(f"Player {human_index} position out of court bounds at frame {frame_index}")

                    # Resize the court image to match the height of the original frame if necessary
                    if court_image.shape[0] != original_frame.shape[0]:
                        court_image = cv2.resize(court_image, (court_width_px, original_frame.shape[0]))

                    # Combine the original frame and the court image side by side
                    combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

                    # Place the original frame on the left
                    combined_frame[:original_frame.shape[0], :original_frame.shape[1]] = original_frame

                    # Place the court image on the right
                    combined_frame[:court_image.shape[0], original_frame.shape[1]:] = court_image

                    # Write the combined frame to the video
                    output_video.write(combined_frame)

                logging.info(f'Visualization video created: {output_video_path}')
    except Exception as e:
        logging.exception("An error occurred during processing.")
        sys.exit(1)


if __name__ == "__main__":
    main()