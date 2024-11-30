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


def compute_iou(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (tuple): Bounding box A (x, y, w, h).
        boxB (tuple): Bounding box B (x, y, w, h).

    Returns:
        float: IoU value.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) != 0 else 0
    return iou


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
    if direction_world[2] == 0:
        logging.error("Direction vector's Z component is zero, cannot proceed with backprojection.")
        raise ValueError("Direction vector's Z component is zero.")

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

            # Step 2: Read Pose Data
            df_pose = read_pose_data(pose_csv)

            # Initialize tracking variables
            tracks = []
            track_id_counter = 0
            max_track_age = 5  # Maximum frames to keep a track without updates
            max_distance = 1.0  # Maximum allowed distance (in meters) for matching

            # Define typical hip height (approx. 0.9 meters for average adult)
            hip_height = 0.9  # meters

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

            output_video_size = (output_width, output_height)

            with VideoWriterContext(output_video_path, fourcc, fps, output_video_size) as output_video:

                # Initialize variables to store player positions
                player_positions = {}

                # Process frames
                for frame_idx in tqdm(range(frame_count), desc='Processing frames'):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Get pose data for the current frame
                    frame_pose_data = df_pose[df_pose['frame_index'] == frame_idx]

                    # List to hold detections for this frame
                    detections = []

                    # For each human detected in pose data
                    humans = frame_pose_data['human_index'].unique()
                    for human_idx in humans:
                        human_data = frame_pose_data[frame_pose_data['human_index'] == human_idx]

                        # Get keypoints
                        keypoints = []
                        for joint_idx in range(17):  # Assuming 17 keypoints
                            joint = human_data[human_data['joint_index'] == joint_idx]
                            if not joint.empty:
                                x = joint['x'].values[0]
                                y = joint['y'].values[0]
                                conf = joint['confidence'].values[0]
                                keypoints.append((x, y, conf))
                            else:
                                keypoints.append((0, 0, 0))  # Placeholder for missing keypoints

                        # Compute player position from ankles and hips
                        # Process ankle and hip points
                        ankle_points = [keypoints[15], keypoints[16]]  # Ankles
                        hip_points = [keypoints[11], keypoints[12]]    # Hips

                        # Compute average ankle position
                        ankle_coords = [(kp[0], kp[1]) for kp in ankle_points if kp[2] > 0.5]
                        if ankle_coords:
                            ankle_point = np.mean(ankle_coords, axis=0)
                        else:
                            ankle_point = None

                        # Compute average hip position
                        hip_coords = [(kp[0], kp[1]) for kp in hip_points if kp[2] > 0.5]
                        if hip_coords:
                            hip_point = np.mean(hip_coords, axis=0)
                        else:
                            hip_point = None

                        if ankle_point is None and hip_point is None:
                            continue  # Skip if no data

                        world_points_list = []

                        # Process ankle point
                        if ankle_point is not None:
                            try:
                                # Backproject the ankle image point to 3D at ground level
                                world_point_ankle = backproject_to_3d(
                                    ankle_point, camera_matrix, dist_coeffs, R, t, point_height=0.0)
                                world_points_list.append(world_point_ankle)
                            except Exception as e:
                                logging.error(f"Error backprojecting ankle point at frame {frame_idx}: {e}")
                                continue

                        # Process hip point
                        if hip_point is not None:
                            try:
                                # Backproject the hip image point to 3D at hip height
                                world_point_hip = backproject_to_3d(
                                    hip_point, camera_matrix, dist_coeffs, R, t, point_height=hip_height)
                                # Project the hip point vertically down to the ground plane
                                world_point_hip_ground = project_point_to_ground(world_point_hip)
                                world_points_list.append(world_point_hip_ground)
                            except Exception as e:
                                logging.error(f"Error backprojecting hip point at frame {frame_idx}: {e}")
                                continue

                        # Average the world points
                        if world_points_list:
                            world_point = np.mean(world_points_list, axis=0)
                            X_world, Y_world = world_point[0], world_point[1]
                        else:
                            continue  # No valid world points

                        detection = {
                            'position': np.array([X_world, Y_world]),
                            'keypoints': keypoints
                        }
                        detections.append(detection)

                    # Update tracks with detections
                    matched_indices = set()
                    for track in tracks:
                        # Find the closest detection
                        min_distance = float('inf')
                        best_detection_idx = None
                        for idx, det in enumerate(detections):
                            distance = np.linalg.norm(track['position'] - det['position'])
                            if distance < min_distance and distance < max_distance:
                                min_distance = distance
                                best_detection_idx = idx
                        if best_detection_idx is not None:
                            # Update track with new detection
                            track['position'] = detections[best_detection_idx]['position']
                            track['age'] = 0
                            track['keypoints'] = detections[best_detection_idx]['keypoints']
                            matched_indices.add(best_detection_idx)
                        else:
                            # No matching detection, increase track age
                            track['age'] += 1

                    # Remove old tracks
                    tracks = [track for track in tracks if track['age'] <= max_track_age]

                    # Create new tracks for unmatched detections
                    unmatched_detections = [det for idx, det in enumerate(detections) if idx not in matched_indices]
                    for det in unmatched_detections:
                        new_track = {
                            'id': track_id_counter,
                            'position': det['position'],
                            'age': 0,
                            'keypoints': det['keypoints']
                        }
                        track_id_counter += 1
                        tracks.append(new_track)

                    # Create a copy of the court image
                    court_image = court_image_template.copy()

                    # Plot player positions
                    for track in tracks:
                        tracker_id = track['id']
                        X_world, Y_world = track['position']

                        # Convert from court coordinates (meters) to pixel coordinates on the court image
                        # Adjust for margin
                        pixel_x = int((X_world + margin_m) * court_scale)
                        pixel_y = int((Y_world + margin_m) * court_scale)  # Adjusted mapping

                        # Determine if the player is on the bottom or upper half
                        color = (0, 0, 255) if Y_world < (court_length_m / 2) else (0, 255, 0)

                        # Plot the player's position on the court image
                        cv2.circle(court_image, (pixel_x, pixel_y), 5, color, -1)
                        # Label the player with the tracker ID
                        cv2.putText(court_image, f"ID: {tracker_id}", (pixel_x + 5, pixel_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Resize the court image to match the height of the original frame if necessary
                    if court_image.shape[0] != frame.shape[0]:
                        court_image = cv2.resize(court_image, (court_width_px, frame.shape[0]))

                    # Combine the original frame and the court image side by side
                    combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

                    # Place the original frame on the left
                    combined_frame[:frame.shape[0], :frame.shape[1]] = frame

                    # Place the court image on the right
                    combined_frame[:court_image.shape[0], frame.shape[1]:] = court_image

                    # Label the frame number in the top-left corner
                    frame_label_position = (10, 30)  # Adjust the position as needed
                    cv2.putText(combined_frame, f"Frame: {frame_idx}", frame_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Write the combined frame to the video
                    output_video.write(combined_frame)

                logging.info(f'Visualization video created: {output_video_path}')
    except Exception as e:
        logging.exception("An error occurred during processing.")
        sys.exit(1)


if __name__ == "__main__":
    main()