import sys
import os
import numpy as np
import cv2
import pandas as pd
import argparse
import logging
import csv
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    ], dtype=np.float32)
    return world_points


def read_court_points(court_csv):
    """
    Reads the 2D court image points from the CSV file.
    Args:
        court_csv (str): Path to the court landmarks CSV file.
    Returns:
        np.ndarray: An array of image points in pixels (X, Y).
    """
    try:
        df = pd.read_csv(court_csv)
        # Ensure required columns are present
        if not {'Point', 'X', 'Y'}.issubset(df.columns):
            logging.error(f"Required columns are missing in {court_csv}.")
            sys.exit(1)
        # Use only the four corner points
        corner_points = df[df['Point'].isin(['P1', 'P2', 'P3', 'P4'])]
        image_points = corner_points[['X', 'Y']].values.astype(np.float32)
        return image_points
    except FileNotFoundError:
        logging.error(f"File {court_csv} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"File {court_csv} is empty.")
        sys.exit(1)


def calibrate_camera(world_points, image_points, camera_matrix):
    """
    Performs camera pose estimation to obtain extrinsic parameters.
    Args:
        world_points (np.ndarray): Array of world points.
        image_points (np.ndarray): Array of corresponding image points.
        camera_matrix (np.ndarray): The camera intrinsic matrix.
    Returns:
        tuple: rvec, tvec, dist_coeffs
    """
    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))
    # Solve the PnP problem to get rotation and translation vectors
    ret, rvec, tvec = cv2.solvePnP(
        world_points, image_points, camera_matrix, dist_coeffs)
    if not ret:
        logging.error("Camera pose estimation failed.")
        sys.exit(1)
    return rvec, tvec, dist_coeffs


def read_pose_data(pose_csv):
    """
    Reads the pose estimation data from the CSV file.
    Args:
        pose_csv (str): Path to the pose estimation CSV file.
    Returns:
        pd.DataFrame: Pose data DataFrame.
    """
    try:
        df = pd.read_csv(pose_csv)
        required_columns = {'frame_index', 'human_index', 'joint_index', 'x', 'y', 'confidence'}
        if not required_columns.issubset(df.columns):
            logging.error(f"Required columns are missing in {pose_csv}.")
            sys.exit(1)
        return df
    except FileNotFoundError:
        logging.error(f"File {pose_csv} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"File {pose_csv} is empty.")
        sys.exit(1)


def compute_player_locations(df, min_confidence_threshold=0.5):
    """
    Computes player positions based on ankles and hips.
    Args:
        df (pd.DataFrame): Pose data DataFrame.
        min_confidence_threshold (float): Minimum confidence to consider limbs visible.
    Returns:
        dict: A dictionary with frame_index as keys and lists of player data.
    """
    results = {}
    df = df.sort_values(by='frame_index')
    grouped = df.groupby(['frame_index', 'human_index'])
    for (frame_index, human_index), joints in grouped:
        if frame_index not in results:
            results[frame_index] = []
        player_data = {'human_index': human_index, 'joints': {}}
        for joint_name, joint_indices in [('ankle', [15, 16]), ('hip', [11, 12])]:
            joint_joints = joints[joints['joint_index'].isin(joint_indices)]
            high_confidence = joint_joints[joint_joints['confidence'] > min_confidence_threshold]
            if len(high_confidence) >= 1:
                x = high_confidence['x'].mean()
                y = high_confidence['y'].mean()
                player_data['joints'][joint_name] = np.array([x, y], dtype=np.float32)
        if player_data['joints']:
            results[frame_index].append(player_data)
    return results


def backproject_to_ground(image_point, camera_matrix, dist_coeffs, R, t):
    """
    Backprojects an image point onto the ground plane (Z=0).
    Args:
        image_point (np.ndarray): The image point (u, v).
        camera_matrix (np.ndarray): The camera matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        R (np.ndarray): The rotation matrix.
        t (np.ndarray): The translation vector.
    Returns:
        np.ndarray or None: The world point (X, Y, Z=0) on the ground plane, or None if cannot compute.
    """
    # Undistort the image point
    undistorted_point = cv2.undistortPoints(
        np.array([[image_point]], dtype=np.float32), camera_matrix, dist_coeffs, P=None)

    x = undistorted_point[0, 0, 0]
    y = undistorted_point[0, 0, 1]

    point_cam = np.array([x, y, 1.0])
    origin_world = -R.T @ t
    direction_world = R.T @ point_cam

    if abs(direction_world[2]) < 1e-6:
        return None

    s = -origin_world[2] / direction_world[2]
    world_point = origin_world + s * direction_world

    return world_point


def track_players(frames_data):
    """
    Tracks players across frames, ensuring only two players are considered.
    Args:
        frames_data (dict): Dictionary containing player data for each frame.
    """
    previous_positions = None  # Stores positions from the previous frame

    for frame_index in sorted(frames_data.keys()):
        current_players = frames_data[frame_index]
        detections = []
        for player_data in current_players:
            world_pos = player_data.get('world_position')
            if world_pos is not None:
                detections.append((player_data, world_pos[:2]))

        if previous_positions is None:
            # Initialize tracked IDs
            for idx, (player_data, _) in enumerate(detections[:2]):  # Only consider first two detections
                player_data['tracked_id'] = idx
            previous_positions = {player_data['tracked_id']: pos for player_data, pos in detections[:2]}
        else:
            # Match detections to previous positions
            prev_ids = list(previous_positions.keys())
            prev_positions = np.array(list(previous_positions.values()))
            curr_positions = np.array([pos for _, pos in detections])

            # Compute cost matrix
            cost_matrix = np.linalg.norm(prev_positions[:, np.newaxis] - curr_positions, axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned = set()
            for r, c in zip(row_ind, col_ind):
                if r < len(prev_ids) and c < len(detections):
                    tracked_id = prev_ids[r]
                    player_data, pos = detections[c]
                    player_data['tracked_id'] = tracked_id
                    previous_positions[tracked_id] = pos
                    assigned.add(c)

            # Handle unassigned detections (new or extra poses)
            unassigned_detections = [i for i in range(len(detections)) if i not in assigned]
            for i in unassigned_detections:
                if len(previous_positions) < 2:
                    # Assign new tracked ID
                    tracked_id = max(previous_positions.keys()) + 1
                    player_data, pos = detections[i]
                    player_data['tracked_id'] = tracked_id
                    previous_positions[tracked_id] = pos
                else:
                    # Extra detection, ignore
                    continue

            # Remove missing players
            previous_positions = {player_data['tracked_id']: pos for player_data, pos in detections if 'tracked_id' in player_data}


class KalmanFilter:
    """
    A simple Kalman Filter for tracking position and velocity in 2D.
    """

    def __init__(self, dt=1.0, process_noise_std=1.0, measurement_noise_std=1.0):
        """
        Initializes the Kalman Filter.
        Args:
            dt (float): Time interval between measurements.
            process_noise_std (float): Standard deviation of the process noise.
            measurement_noise_std (float): Standard deviation of the measurement noise.
        """
        # Time interval
        self.dt = dt

        # State vector: [x_position, y_position, x_velocity, y_velocity]
        self.x = np.zeros((4, 1))

        # State transition matrix
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Control matrix
        self.B = np.zeros((4, 1))

        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process noise covariance
        q = process_noise_std ** 2
        self.Q = q * np.eye(4)

        # Measurement noise covariance
        r = measurement_noise_std ** 2
        self.R = r * np.eye(2)

        # Initial covariance matrix
        self.P = np.eye(4)

    def predict(self):
        """
        Predicts the next state and updates the covariance matrix.
        """
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        """
        Updates the state vector with a new measurement.
        Args:
            z (np.ndarray): The measurement vector [x_position, y_position].
        """
        z = z.reshape(2, 1)
        y = z - self.H @ self.x  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P


def apply_kalman_filter(frames_data):
    """
    Applies Kalman filter to the player positions to reduce jitter.
    Args:
        frames_data (dict): Dictionary containing player data for each frame.
    """
    # Initialize Kalman filters for each player
    kalman_filters = {}

    for frame_index in sorted(frames_data.keys()):
        players = frames_data[frame_index]
        for player_data in players:
            if 'tracked_id' in player_data and 'world_position' in player_data:
                tracked_id = player_data['tracked_id']
                measurement = player_data['world_position'][:2]  # X_world, Y_world

                if tracked_id not in kalman_filters:
                    # Initialize a new Kalman filter for this player
                    kf = KalmanFilter(dt=1.0, process_noise_std=1.0, measurement_noise_std=1.0)
                    kf.x[:2] = measurement.reshape(2, 1)  # Initialize position
                    kalman_filters[tracked_id] = kf
                else:
                    kf = kalman_filters[tracked_id]

                # Predict the next state
                kf.predict()

                # Update with the current measurement
                kf.update(measurement)

                # Update the player data with the filtered position
                filtered_position = kf.x[:2].flatten()
                player_data['world_position'][:2] = filtered_position


def draw_court(court_image, court_scale):
    """
    Draws the badminton court on the court image.
    """
    court_length_m = 13.4
    court_width_m = 6.1
    margin_m = 2.0
    length_with_margin = court_length_m + 2 * margin_m
    width_with_margin = court_width_m + 2 * margin_m

    left = int(margin_m * court_scale)
    right = int((margin_m + court_width_m) * court_scale)
    top = int(margin_m * court_scale)
    bottom = int((margin_m + court_length_m) * court_scale)

    # Draw outer boundaries
    cv2.rectangle(court_image, (left, top), (right, bottom), (255, 255, 255), 2)

    # Draw the net at half the length of the court
    net_y = int((margin_m + court_length_m / 2) * court_scale)
    cv2.line(court_image, (left, net_y), (right, net_y), (255, 255, 255), 2)


def main():
    """
    Main function to process, compute player positions, and visualize.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compute player positions on badminton court and visualize.')
    parser.add_argument('video_path', type=str, help='Path to input video file.')
    parser.add_argument('--output_video', type=str, help='Path to output video file.')
    args = parser.parse_args()

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
        output_video_path = f'result/{video_name_without_ext}_positions.mp4'

    # Ensure the 'result' directory exists
    os.makedirs('result', exist_ok=True)

    # Read court world points and image points
    world_points = define_world_points()
    image_points = read_court_points(court_csv)

    if len(world_points) != len(image_points):
        logging.error("The number of world points and image points must be equal for calibration.")
        sys.exit(1)

    # Open the video to get the image size and FPS
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file {args.video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Assume approximate camera intrinsic parameters
    focal_length = original_width  # Approximation
    center = (original_width / 2, original_height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float32)

    # Perform camera pose estimation
    rvec, tvec, dist_coeffs = calibrate_camera(world_points, image_points, camera_matrix)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    # Read pose data
    df_pose = read_pose_data(pose_csv)
    if df_pose.empty:
        logging.error("Pose CSV file is empty or has no data.")
        sys.exit(1)

    # Compute player locations
    frames_data = compute_player_locations(df_pose)

    # Process frames data to compute world coordinates
    for frame_index, players in frames_data.items():
        for player_data in players:
            joints = player_data['joints']
            # Backproject joints onto the ground plane
            if 'ankle' in joints:
                image_point = joints['ankle']
            elif 'hip' in joints:
                image_point = joints['hip']
            else:
                continue

            world_position = backproject_to_ground(
                image_point, camera_matrix, dist_coeffs, R, t)
            if world_position is not None:
                player_data['world_position'] = world_position

    # Track players and filter extra poses
    track_players(frames_data)

    # Apply Kalman filter to smooth player positions
    apply_kalman_filter(frames_data)

    # Save player positions to CSV
    output_rows = []
    for frame_index, players in frames_data.items():
        for player_data in players:
            if 'tracked_id' not in player_data:
                continue
            tracked_id = player_data['tracked_id']
            world_position = player_data['world_position']
            x_img, y_img = player_data['joints'].get('ankle', player_data['joints'].get('hip', (0, 0)))
            X_world, Y_world, Z_world = world_position
            output_rows.append([frame_index, tracked_id, x_img, y_img, X_world, Y_world, Z_world])

    output_csv_path = os.path.join('result', f'{video_name_without_ext}_positions.csv')
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['frame_index', 'tracked_id', 'img_x', 'img_y', 'world_X', 'world_Y', 'world_Z'])
        writer.writerows(output_rows)

    logging.info(f'Player positions saved to {output_csv_path}')

    # Visualization
    # Define court dimensions
    court_length_m = 13.4
    court_width_m = 6.1
    margin_m = 2.0
    court_scale = 50  # pixels per meter

    court_img_height = int((court_length_m + 2 * margin_m) * court_scale)
    court_img_width = int((court_width_m + 2 * margin_m) * court_scale)
    court_image_template = np.zeros((court_img_height, court_img_width, 3), dtype=np.uint8)
    draw_court(court_image_template, court_scale)

    # Create a VideoWriter object
    output_width = original_width + court_img_width
    output_height = max(original_height, court_img_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    if not output_video.isOpened():
        logging.error(f"Cannot open video writer with filename {output_video_path}")
        sys.exit(1)

    # Assign colors to tracked IDs
    colors = [(0, 255, 0), (0, 0, 255)]  # Green and Red for two players
    tracked_id_to_color = {i: colors[i % 2] for i in range(2)}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
    for frame_index in tqdm(range(frame_count), desc='Processing frames'):
        ret, frame = cap.read()
        if not ret:
            break

        # Create a copy of the court image
        court_image = court_image_template.copy()

        # Get frame data
        frame_data = frames_data.get(frame_index, [])

        # Plot player positions
        for player_data in frame_data:
            if 'tracked_id' not in player_data:
                continue
            tracked_id = player_data['tracked_id']
            color = tracked_id_to_color.get(tracked_id, (255, 255, 255))  # Default to white if not found
            world_position = player_data.get('world_position')
            if world_position is None:
                continue

            X_world, Y_world = world_position[0], world_position[1]

            # Convert from court coordinates (meters) to pixel coordinates on the court image
            pixel_x = int((X_world + margin_m) * court_scale)
            pixel_y = int((court_length_m + margin_m - Y_world) * court_scale)  # Adjust Y to invert

            # Plot the player's position on the court image
            cv2.circle(court_image, (pixel_x, pixel_y), 5, color, -1)
            # Optionally, draw the player ID
            cv2.putText(court_image, f"P{tracked_id}", (pixel_x - 15, pixel_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw a circle on the original frame at the ankle position
            x_img, y_img = player_data['joints'].get('ankle', player_data['joints'].get('hip', (0, 0)))
            cv2.circle(frame, (int(x_img), int(y_img)), 5, color, -1)
            cv2.putText(frame, f"P{tracked_id}", (int(x_img) - 15, int(y_img) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Combine original frame and court image
        combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        combined_frame[:original_height, :original_width] = frame
        combined_frame[:court_image.shape[0], original_width:] = court_image

        # Write the frame to the video
        output_video.write(combined_frame)

    output_video.release()
    cap.release()
    logging.info(f'Visualization video created: {output_video_path}')


if __name__ == "__main__":
    main()