import sys
import os
import numpy as np
import cv2
import pandas as pd
import argparse
import logging
from tqdm import tqdm
import csv

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


def read_pose_data(pose_csv):
    """
    Reads the pose estimation data from the CSV file.

    Args:
        pose_csv (str): Path to the pose estimation CSV file.

    Returns:
        pd.DataFrame: Pose data DataFrame.

    Raises:
        FileNotFoundError: If the pose CSV file is not found.
        pd.errors.EmptyDataError: If the pose CSV file is empty.
    """
    try:
        df = pd.read_csv(pose_csv)
        return df
    except FileNotFoundError as e:
        logging.error(f"File {pose_csv} not found.")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"File {pose_csv} is empty.")
        raise e


def low_pass_filter(current_value, previous_value, alpha):
    """
    Applies a low-pass filter to the given values to smooth out spikes.

    Args:
        current_value (float): The current value (e.g., velocity component).
        previous_value (float): The previous filtered value.
        alpha (float): The smoothing factor.

    Returns:
        float: The filtered value.
    """
    return alpha * current_value + (1 - alpha) * previous_value


def compute_player_locations(df, base_alpha=0.3, max_prediction_frames=5, min_confidence_threshold=0.5):
    """
    Computes the player's positions based on ankles and hips, applies smoothing, and returns both.

    Args:
        df (pd.DataFrame): Pose data DataFrame.
        base_alpha (float): The base smoothing factor for EMA.
        max_prediction_frames (int): The maximum number of frames to predict the player's position when joints are not visible.
        min_confidence_threshold (float): Minimum confidence threshold to consider limbs visible.

    Returns:
        dict: A dictionary with keys as frame_index and values as dictionaries containing player detections.
    """
    results = {}
    previous_positions = {}  # Store past positions for each player and joint
    frames_without_joints = {}  # Track missing frames for joints

    # Sort the dataframe by frame_index to process frames in order
    df = df.sort_values(by='frame_index')

    grouped = df.groupby(['frame_index', 'human_index'])
    for (frame_index, human_index), joints in grouped:
        if frame_index not in results:
            results[frame_index] = []
        player_data = {'human_index': human_index, 'joints': {}}
        for joint_name, joint_indices in [('ankle', [15, 16]), ('hip', [11, 12])]:
            joint_joints = joints[joints['joint_index'].isin(joint_indices)]
            high_confidence = joint_joints[joint_joints['confidence'] > min_confidence_threshold]

            joint_prev_key = (human_index, joint_name)

            if len(high_confidence) >= 1:
                # At least one joint is visible, compute mean position
                x = high_confidence['x'].mean()
                y = high_confidence['y'].mean()

                # Apply smoothing
                if joint_prev_key in previous_positions:
                    prev_x, prev_y = previous_positions[joint_prev_key]
                    smoothed_x = low_pass_filter(x, prev_x, base_alpha)
                    smoothed_y = low_pass_filter(y, prev_y, base_alpha)
                else:
                    smoothed_x, smoothed_y = x, y  # No previous position, start with current

                # Store the smoothed position
                previous_positions[joint_prev_key] = (smoothed_x, smoothed_y)
                player_data['joints'][joint_name] = np.array([smoothed_x, smoothed_y], dtype=np.float32)

                # Reset the missing frames count
                frames_without_joints[joint_prev_key] = 0
            else:
                # Joint not detected
                frames_without_joints[joint_prev_key] = frames_without_joints.get(joint_prev_key, 0) + 1
                if frames_without_joints[joint_prev_key] <= max_prediction_frames:
                    # Predict position based on previous positions
                    if joint_prev_key in previous_positions:
                        smoothed_x, smoothed_y = previous_positions[joint_prev_key]
                        player_data['joints'][joint_name] = np.array([smoothed_x, smoothed_y], dtype=np.float32)
                else:
                    logging.warning(f"Player {human_index} joint {joint_name} missing for too long at frame {frame_index}.")

        if player_data['joints']:
            results[frame_index].append(player_data)

    return results


def draw_court(image, court_scale, court_length_m_with_margin, court_width_m_with_margin, margin_m, court_length_m):
    """
    Draws the court lines on the schematic court image.

    Args:
        image (np.ndarray): The court image to draw on.
        court_scale (float): Scale factor from meters to pixels.
        court_length_m_with_margin (float): The length of the court including margins (in meters).
        court_width_m_with_margin (float): The width of the court including margins (in meters).
        margin_m (float): The margin added around the court (in meters).
        court_length_m (float): The actual length of the court (without margins, in meters).
    """
    # Compute the positions of the court lines in pixels
    left = int(margin_m * court_scale)
    right = int((court_width_m_with_margin - margin_m) * court_scale)
    top = int(margin_m * court_scale)
    bottom = int((court_length_m_with_margin - margin_m) * court_scale)

    # Draw outer boundaries
    cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), 2)

    # Draw the net at half the length of the court (considering margins)
    net_y = int((margin_m + court_length_m / 2) * court_scale)
    cv2.line(image, (left, net_y), (right, net_y), (255, 255, 255), 2)


def backproject_to_3d(image_point, camera_matrix, dist_coeffs, R, t, point_height):
    """
    Backprojects an image point to a 3D point at the given height.

    Args:
        image_point (np.ndarray): The image point (u, v).
        camera_matrix (np.ndarray): The camera matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        R (np.ndarray): The rotation matrix.
        t (np.ndarray): The translation vector.
        point_height (float): The height of the point above the ground in world coordinates.

    Returns:
        np.ndarray: The world point (X, Y, Z) at the specified height.
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

    # Compute scale factor s for the specified height
    s = (point_height - origin_world[2]) / direction_world[2]
    world_point = origin_world + s * direction_world

    return world_point


def project_point_to_ground(world_point):
    """
    Projects a 3D point to the ground plane by setting z=0.

    Args:
        world_point (np.ndarray): The 3D world point (X, Y, Z).

    Returns:
        np.ndarray: The projected 3D world point on the ground plane (X, Y, 0).
    """
    return np.array([world_point[0], world_point[1], 0.0])


def compute_average_offset(frames_data):
    """
    Computes the average offset between ankle and hip projections onto the ground plane.

    Args:
        frames_data (dict): The dictionary containing player data for each frame.

    Returns:
        np.ndarray: The average offset (dx, dy) to correct hip projections.
    """
    offsets = []
    for frame_data in frames_data.values():
        for player_data in frame_data:
            ankle_world = player_data.get('ankle_world')
            hip_world = player_data.get('hip_world')
            if ankle_world is not None and hip_world is not None:
                offset = ankle_world[:2] - hip_world[:2]
                offsets.append(offset)
    if offsets:
        average_offset = np.mean(offsets, axis=0)
        logging.info(f"Average offset between ankle and hip projections: {average_offset}")
        return average_offset
    else:
        return np.array([0.0, 0.0])


def track_players(frames_data, max_distance=2.0):
    """
    Tracks players across frames using a distance-based tracking system.

    Args:
        frames_data (dict): The dictionary containing player data for each frame.
        max_distance (float): Maximum distance to consider for tracking (in meters).

    Modifies:
        frames_data: Updates the 'tracked_id' for each player in frames_data.
    """
    next_tracked_id = 0
    tracks = {}  # tracked_id: last position

    for frame_index in sorted(frames_data.keys()):
        current_players = frames_data[frame_index]
        unmatched_tracks = set(tracks.keys())
        for player_data in current_players:
            min_distance = float('inf')
            matched_id = None
            player_pos = None

            # Use ankle_world if available, else use hip_world
            if player_data.get('ankle_world') is not None:
                player_pos = player_data['ankle_world'][:2]
            elif player_data.get('hip_world') is not None:
                player_pos = player_data['hip_world'][:2]
            else:
                continue  # No valid position

            # Find the closest track
            for tracked_id, last_pos in tracks.items():
                distance = np.linalg.norm(player_pos - last_pos)
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    matched_id = tracked_id

            if matched_id is not None:
                # Assign the matched tracked_id
                player_data['tracked_id'] = matched_id
                tracks[matched_id] = player_pos  # Update the position
                unmatched_tracks.discard(matched_id)
            else:
                # Start a new track
                player_data['tracked_id'] = next_tracked_id
                tracks[next_tracked_id] = player_pos
                next_tracked_id += 1

        # Remove tracks not matched for this frame
        for unmatched_id in unmatched_tracks:
            del tracks[unmatched_id]


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

        # Ensure the 'result' directory exists
        os.makedirs('result', exist_ok=True)

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
            df_pose = read_pose_data(pose_csv)
            frames_data = compute_player_locations(df_pose)

            # Define typical hip height (approx. 0.9 meters for average adult)
            hip_height = 0.9  # meters

            # Process frames data to compute world coordinates and apply corrections
            for frame_index, players in frames_data.items():
                for player_data in players:
                    joints = player_data['joints']
                    # Backproject joints to 3D and project to ground plane
                    if 'ankle' in joints:
                        ankle_point = joints['ankle']
                        ankle_world = backproject_to_3d(
                            ankle_point, camera_matrix, dist_coeffs, R, t, point_height=0.0)
                        player_data['ankle_world'] = ankle_world
                    else:
                        player_data['ankle_world'] = None

                    if 'hip' in joints:
                        hip_point = joints['hip']
                        hip_world = backproject_to_3d(
                            hip_point, camera_matrix, dist_coeffs, R, t, point_height=hip_height)
                        # Project to ground plane
                        hip_world_ground = project_point_to_ground(hip_world)
                        player_data['hip_world'] = hip_world_ground
                    else:
                        player_data['hip_world'] = None

            # Compute average offset between ankle and hip projections
            average_offset = compute_average_offset(frames_data)

            # Apply the average offset to hip projections
            for frame_index, players in frames_data.items():
                for player_data in players:
                    if player_data['hip_world'] is not None:
                        player_data['hip_world'][:2] += average_offset

            # Implement tracking to maintain consistent player IDs
            track_players(frames_data)

            # Save player positions to CSV
            output_csv_path = os.path.join('result', f'{video_name_without_ext}_position.csv')
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['frame_index', 'tracked_id', 'joint_type', 'img_x', 'img_y', 'world_X', 'world_Y', 'world_Z'])
                for frame_index, players in frames_data.items():
                    for player_data in players:
                        tracked_id = player_data.get('tracked_id', -1)
                        joints = player_data['joints']
                        for joint_type in ['ankle', 'hip']:
                            joint_point = joints.get(joint_type)
                            if joint_point is None:
                                continue  # Skip if no data
                            if joint_type == 'ankle':
                                world_point = player_data['ankle_world']
                            else:
                                world_point = player_data['hip_world']
                            X_world, Y_world, Z_world = world_point
                            x_img, y_img = joint_point
                            writer.writerow([frame_index, tracked_id, joint_type, x_img, y_img, X_world, Y_world, Z_world])

            logging.info(f'Player positions saved to {output_csv_path}')

            # Define the margins in meters
            margin_m = 2  # meters

            # Adjust the court dimensions
            court_length_m = 13.4  # meters
            court_width_m = 6.1    # meters
            court_length_m_with_margin = court_length_m + 2 * margin_m
            court_width_m_with_margin = court_width_m + 2 * margin_m

            # Define the size of the court image
            court_height_px = 720  # Adjust this value as needed
            court_scale = court_height_px / court_length_m_with_margin  # pixels per meter
            court_width_px = int(court_width_m_with_margin * court_scale)

            # Create a blank court image template
            court_image_template = np.zeros((court_height_px, court_width_px, 3), dtype=np.uint8)

            # Draw the court lines on the template
            draw_court(
                court_image_template,
                court_scale,
                court_length_m_with_margin,
                court_width_m_with_margin,
                margin_m,
                court_length_m
            )

            # Adjust output video properties
            output_width = original_width + court_width_px
            output_height = max(original_height, court_height_px)

            # Create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

            # Define colors for each player
            player_colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (255, 165, 0),  # Orange
                (128, 0, 128),  # Purple
                (0, 128, 128),  # Teal
                (128, 128, 0),  # Olive
                # Add more colors if needed
            ]

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
                    frame_data = frames_data.get(frame_index, [])

                    # Plot player positions
                    for player_data in frame_data:
                        tracked_id = player_data.get('tracked_id', -1)
                        color = player_colors[tracked_id % len(player_colors)]
                        joints = player_data['joints']

                        for joint_type in ['ankle', 'hip']:
                            joint_point = joints.get(joint_type)
                            if joint_point is None:
                                continue  # Skip if no data
                            if joint_type == 'ankle':
                                world_point = player_data['ankle_world']
                            else:
                                world_point = player_data['hip_world']

                            X_world, Y_world = world_point[0], world_point[1]

                            # Convert from court coordinates (meters) to pixel coordinates on the court image
                            # Adjust for margin
                            pixel_x = int((X_world + margin_m) * court_scale)
                            pixel_y = int((Y_world + margin_m) * court_scale)  # Adjusted mapping

                            # Plot the player's position on the court image
                            cv2.circle(court_image, (pixel_x, pixel_y), 5, color, -1)

                    # Resize the court image to match the height of the original frame if necessary
                    if court_image.shape[0] != original_frame.shape[0]:
                        court_image = cv2.resize(court_image, (court_width_px, original_frame.shape[0]))

                    # Combine the original frame and the court image side by side
                    combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

                    # Place the original frame on the left
                    combined_frame[:original_frame.shape[0], :original_frame.shape[1]] = original_frame

                    # Place the court image on the right
                    combined_frame[:court_image.shape[0], original_frame.shape[1]:] = court_image

                    # Label the frame number in the top-left corner
                    frame_label_position = (10, 30)  # Adjust the position as needed
                    cv2.putText(combined_frame, f"Frame: {frame_index}", frame_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Write the combined frame to the video
                    output_video.write(combined_frame)

                logging.info(f'Visualization video created: {output_video_path}')
    except Exception as e:
        logging.exception("An error occurred during processing.")
        sys.exit(1)


if __name__ == "__main__":
    main()