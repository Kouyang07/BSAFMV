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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HIP_HEIGHT = 1.0  # in meters, approximate average hip height

def define_world_points():
    """
    Defines 3D world points of the badminton court (the corner points).
    """
    world_points = np.array([
        [0, 0, 0],
        [0, 13.4, 0],
        [6.1, 13.4, 0],
        [6.1, 0, 0]
    ], dtype=np.float32)
    return world_points

def read_court_points(court_csv):
    """
    Reads the 2D court image corner points from CSV.
    """
    try:
        df = pd.read_csv(court_csv)
        if not {'Point', 'X', 'Y'}.issubset(df.columns):
            logging.error("Missing required columns in court CSV.")
            sys.exit(1)
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
    Perform camera pose estimation (PnP).
    """
    dist_coeffs = np.zeros((4,1))
    ret, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)
    if not ret:
        logging.error("Camera pose estimation failed.")
        sys.exit(1)
    return rvec, tvec, dist_coeffs

def read_pose_data(pose_csv):
    """
    Reads the pose estimation CSV containing frame_index, human_index, joint_index, x, y, confidence.
    """
    try:
        df = pd.read_csv(pose_csv)
        required_cols = {'frame_index', 'human_index', 'joint_index', 'x', 'y', 'confidence'}
        if not required_cols.issubset(df.columns):
            logging.error("Missing required columns in pose CSV.")
            sys.exit(1)
        return df
    except FileNotFoundError:
        logging.error(f"File {pose_csv} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"File {pose_csv} is empty.")
        sys.exit(1)

def backproject_to_plane(image_point, camera_matrix, dist_coeffs, R, t, plane_height=0.0):
    """
    Backproject an image point onto a horizontal plane at a given height.
    Returns None if projection is invalid.
    """
    undist_pt = cv2.undistortPoints(np.array([[image_point]], dtype=np.float32),
                                    camera_matrix, dist_coeffs, P=None)
    x, y = undist_pt[0,0]
    point_cam = np.array([x, y, 1.0])

    # Get camera position and ray direction in world coordinates
    camera_pos = -R.T @ t
    ray_dir = R.T @ point_cam
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    # Check if ray is parallel to plane
    if abs(ray_dir[2]) < 1e-9:
        return None

    # Calculate intersection with plane
    t = (plane_height - camera_pos[2]) / ray_dir[2]
    if t < 0:  # Check if intersection is behind camera
        return None

    world_point = camera_pos + t * ray_dir
    return world_point

def project_hip_to_ground(hip_point, camera_matrix, dist_coeffs, R, t):
    """
    Projects hip point to ground plane using geometric relationship.
    Returns the ground position accounting for perspective projection.
    """
    # Get camera position in world coordinates
    camera_pos = -R.T @ t

    # Project hip point to hip-height plane first
    hip_plane = backproject_to_plane(hip_point, camera_matrix, dist_coeffs, R, t,
                                     plane_height=HIP_HEIGHT)

    if hip_plane is None:
        return None

    # Vector from camera to hip point
    cam_to_hip = hip_plane - camera_pos

    # Calculate the scaling factor to project to ground
    # Using similar triangles relationship
    scale = -camera_pos[2] / (hip_plane[2] - camera_pos[2])

    # Project to ground using the scaling factor
    ground_point = camera_pos + scale * cam_to_hip
    ground_point[2] = 0  # Ensure point is exactly on ground plane

    return ground_point

def compute_player_locations(df, min_confidence=0.5):
    """
    Extracts hip and ankle information from the pose dataframe.
    """
    results = {}
    df = df.sort_values(by='frame_index')
    grouped = df.groupby(['frame_index', 'human_index'])

    for (frame_idx, human_idx), joints in grouped:
        if frame_idx not in results:
            results[frame_idx] = []
        player_data = {'human_index': human_idx, 'joints': {}}

        # Left ankle (joint 15) and right ankle (joint 16)
        left_ankle = joints[(joints['joint_index'] == 15) & (joints['confidence'] > min_confidence)]
        right_ankle = joints[(joints['joint_index'] == 16) & (joints['confidence'] > min_confidence)]

        if not left_ankle.empty:
            player_data['joints']['left_ankle'] = np.array([left_ankle['x'].mean(), left_ankle['y'].mean()], dtype=np.float32)
        if not right_ankle.empty:
            player_data['joints']['right_ankle'] = np.array([right_ankle['x'].mean(), right_ankle['y'].mean()], dtype=np.float32)

        # Ankle midpoint
        if 'left_ankle' in player_data['joints'] and 'right_ankle' in player_data['joints']:
            la = player_data['joints']['left_ankle']
            ra = player_data['joints']['right_ankle']
            player_data['joints']['ankle_midpoint'] = (la + ra) / 2.0
        elif 'left_ankle' in player_data['joints']:
            player_data['joints']['ankle_midpoint'] = player_data['joints']['left_ankle']
        elif 'right_ankle' in player_data['joints']:
            player_data['joints']['ankle_midpoint'] = player_data['joints']['right_ankle']

        # Hip (average of joints 11 and 12)
        hip_joints = joints[(joints['joint_index'].isin([11, 12])) & (joints['confidence'] > min_confidence)]
        if not hip_joints.empty:
            hx = hip_joints['x'].mean()
            hy = hip_joints['y'].mean()
            player_data['joints']['hip'] = np.array([hx, hy], dtype=np.float32)

        if player_data['joints']:
            results[frame_idx].append(player_data)

    return results

def track_players(frames_data):
    """
    Simple tracking to maintain consistent player IDs across frames.
    """
    previous_positions = None
    for frame_idx in sorted(frames_data.keys()):
        current_players = frames_data[frame_idx]
        detections = []
        for player_data in current_players:
            if 'world_position' in player_data and player_data['world_position'] is not None:
                detections.append((player_data, player_data['world_position'][:2]))

        if previous_positions is None:
            # Assign IDs to first two detections
            for i, (pdata, _) in enumerate(detections[:2]):
                pdata['tracked_id'] = i
            previous_positions = {pdata['tracked_id']: pos for pdata, pos in detections[:2]}
        else:
            prev_ids = list(previous_positions.keys())
            prev_positions = np.array(list(previous_positions.values()))
            curr_positions = np.array([pos for _, pos in detections])

            if len(prev_positions) > 0 and len(curr_positions) > 0:
                cost_matrix = np.linalg.norm(prev_positions[:, None] - curr_positions, axis=2)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                assigned = set()
                for r, c in zip(row_ind, col_ind):
                    if r < len(prev_ids) and c < len(detections):
                        tid = prev_ids[r]
                        pdata, pos = detections[c]
                        pdata['tracked_id'] = tid
                        previous_positions[tid] = pos
                        assigned.add(c)

                unassigned = [i for i in range(len(detections)) if i not in assigned]
                for i in unassigned:
                    if len(previous_positions) < 2:
                        new_id = 0 if not previous_positions else max(previous_positions.keys())+1
                        pdata, pos = detections[i]
                        pdata['tracked_id'] = new_id
                        previous_positions[new_id] = pos
                    # else ignore
                previous_positions = {pdata['tracked_id']: pos for pdata, pos in detections if 'tracked_id' in pdata}

class KalmanFilter:
    """
    Simple Kalman filter for smoothing player positions.
    """
    def __init__(self, dt=1.0, process_noise_std=1.0, measurement_noise_std=1.0):
        self.dt = dt
        self.x = np.zeros((4,1))
        self.A = np.array([[1,0,self.dt,0],
                           [0,1,0,self.dt],
                           [0,0,1,0],
                           [0,0,0,1]])
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]])
        q = process_noise_std**2
        self.Q = q*np.eye(4)
        r = measurement_noise_std**2
        self.R = r*np.eye(2)
        self.P = np.eye(4)

    def predict(self):
        self.x = self.A@self.x
        self.P = self.A@self.P@self.A.T + self.Q

    def update(self, z):
        z = z.reshape(2,1)
        y = z - self.H@self.x
        S = self.H@self.P@self.H.T + self.R
        K = self.P@self.H.T@np.linalg.inv(S)
        self.x = self.x + K@y
        I = np.eye(4)
        self.P = (I - K@self.H)@self.P

def apply_kalman_filter(frames_data):
    kalman_filters = {}
    for frame_idx in sorted(frames_data.keys()):
        players = frames_data[frame_idx]
        for pdata in players:
            if 'tracked_id' in pdata and pdata.get('world_position') is not None:
                tid = pdata['tracked_id']
                meas = pdata['world_position'][:2]
                if tid not in kalman_filters:
                    kf = KalmanFilter()
                    kf.x[:2] = meas.reshape(2,1)
                    kalman_filters[tid] = kf
                else:
                    kf = kalman_filters[tid]
                kf.predict()
                kf.update(meas)
                pdata['world_position'][:2] = kf.x[:2].flatten()

def draw_court(court_image, court_scale):
    court_length_m = 13.4
    court_width_m = 6.1
    margin_m = 2.0
    left = int(margin_m*court_scale)
    right = int((margin_m+court_width_m)*court_scale)
    top = int(margin_m*court_scale)
    bottom = int((margin_m+court_length_m)*court_scale)
    cv2.rectangle(court_image, (left, top), (right, bottom), (255,255,255), 2)
    net_y = int((margin_m + court_length_m/2)*court_scale)
    cv2.line(court_image, (left, net_y), (right, net_y), (255,255,255), 2)

def main():
    parser = argparse.ArgumentParser(description='Compute and visualize player positions using hip offset.')
    parser.add_argument('video_path', type=str, help='Path to input video.')
    parser.add_argument('--output_video', type=str, help='Path to output video.')
    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    pose_csv = os.path.join('result', f'{video_name}_pose.csv')
    court_csv = os.path.join('result', f'{video_name}_court.csv')
    if args.output_video:
        output_video_path = args.output_video
    else:
        output_video_path = f'result/{video_name}_positions.mp4'

    os.makedirs('result', exist_ok=True)

    # Read court data
    world_points = define_world_points()
    image_points = read_court_points(court_csv)

    # Video properties
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logging.error("Cannot open video.")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Camera intrinsics approximation
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float32)
    rvec, tvec, dist_coeffs = calibrate_camera(world_points, image_points, camera_matrix)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    # Pose data
    df_pose = read_pose_data(pose_csv)
    if df_pose.empty:
        logging.error("Pose CSV is empty.")
        sys.exit(1)

    frames_data = compute_player_locations(df_pose)

    # Compute world positions using ankle and hip with offset correction
    for frame_idx, players in frames_data.items():
        for pdata in players:
            j = pdata['joints']

            # If we have ankles, backproject to ground for reference
            if 'ankle_midpoint' in j:
                ankle_ground = backproject_to_plane(j['ankle_midpoint'], camera_matrix, dist_coeffs, R, t, plane_height=0.0)
                if ankle_ground is not None:
                    pdata['ankle_world_position'] = ankle_ground

            # If we have hip, backproject to hip height and ground
            # In the main loop where positions are computed:
            if 'hip' in j:
                hip_point = j['hip']
                ground_position = project_hip_to_ground(hip_point, camera_matrix, dist_coeffs, R, t)

                if ground_position is not None:
                    pdata['world_position'] = ground_position
                elif 'ankle_midpoint' in j:
                    # Fallback to ankle position if hip projection fails
                    ankle_ground = backproject_to_plane(j['ankle_midpoint'], camera_matrix,
                                                        dist_coeffs, R, t, plane_height=0.0)
                    if ankle_ground is not None:
                        pdata['world_position'] = ankle_ground



            # Decide final world position
            if 'hip_world_position_ground' in pdata:
                pdata['world_position'] = pdata['hip_world_position_ground']
            elif 'ankle_world_position' in pdata:
                pdata['world_position'] = pdata['ankle_world_position']

            # Also backproject ankles individually to ground if available:
            if 'left_ankle' in j:
                la_g = backproject_to_plane(j['left_ankle'], camera_matrix, dist_coeffs, R, t, 0.0)
                if la_g is not None:
                    pdata['left_ankle_world'] = la_g
            if 'right_ankle' in j:
                ra_g = backproject_to_plane(j['right_ankle'], camera_matrix, dist_coeffs, R, t, 0.0)
                if ra_g is not None:
                    pdata['right_ankle_world'] = ra_g
            if 'left_ankle_world' in pdata and 'right_ankle_world' in pdata:
                pdata['ankle_midpoint_world'] = (pdata['left_ankle_world'] + pdata['right_ankle_world']) / 2.0
            elif 'left_ankle_world' in pdata:
                pdata['ankle_midpoint_world'] = pdata['left_ankle_world']
            elif 'right_ankle_world' in pdata:
                pdata['ankle_midpoint_world'] = pdata['right_ankle_world']

    # Track players
    track_players(frames_data)

    # Apply Kalman filter
    apply_kalman_filter(frames_data)

    # Save to CSV
    output_csv_path = os.path.join('result', f'{video_name}_positions.csv')
    rows = []
    for frame_idx, players in frames_data.items():
        for pdata in players:
            if 'tracked_id' in pdata:
                tid = pdata['tracked_id']
                if 'world_position' in pdata and pdata['world_position'] is not None:
                    wp = pdata['world_position']
                    # Use ankle_midpoint or hip for image coords
                    if 'ankle_midpoint' in pdata['joints']:
                        x_img, y_img = pdata['joints']['ankle_midpoint']
                    elif 'hip' in pdata['joints']:
                        x_img, y_img = pdata['joints']['hip']
                    else:
                        x_img, y_img = (None, None)
                    rows.append([frame_idx, tid, x_img, y_img, wp[0], wp[1], wp[2]])

    with open(output_csv_path, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['frame_index','tracked_id','img_x','img_y','world_X','world_Y','world_Z'])
        writer.writerows(rows)

    logging.info(f"Player positions saved to {output_csv_path}")

    # Visualization
    court_length_m = 13.4
    court_width_m = 6.1
    margin_m = 2.0
    court_scale = 50
    court_img_h = int((court_length_m + 2*margin_m)*court_scale)
    court_img_w = int((court_width_m + 2*margin_m)*court_scale)
    court_img_template = np.zeros((court_img_h, court_img_w, 3), dtype=np.uint8)
    draw_court(court_img_template, court_scale)

    out_w = w + court_img_w
    out_h = max(h, court_img_h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

    if not output_video.isOpened():
        logging.error(f"Cannot open video writer {output_video_path}")
        sys.exit(1)

    colors = [(0,255,0), (0,0,255)]
    tracked_id_to_color = {i: colors[i%2] for i in range(2)}

    def world_to_court(X, Y):
        px = int((X+margin_m)*court_scale)
        py = int((court_length_m+margin_m - Y)*court_scale)
        return (px, py)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f_idx in tqdm(range(frame_count), desc='Processing frames'):
        ret, frame = cap.read()
        if not ret:
            break
        court_img = court_img_template.copy()
        frame_data = frames_data.get(f_idx, [])

        for pdata in frame_data:
            if 'tracked_id' not in pdata:
                continue
            tid = pdata['tracked_id']
            c = tracked_id_to_color.get(tid, (255,255,255))

            # Draw player position on overhead court
            if 'world_position' in pdata and pdata['world_position'] is not None:
                wp = pdata['world_position']
                wp_px = world_to_court(wp[0], wp[1])
                cv2.circle(court_img, wp_px, 5, c, -1)
                cv2.putText(court_img, f"P{tid}", (wp_px[0]-15, wp_px[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

            # Draw hip height and ground
            if 'hip_world_position_hip_plane' in pdata:
                hh = pdata['hip_world_position_hip_plane']
                hh_px = world_to_court(hh[0], hh[1])
                cv2.circle(court_img, hh_px, 5, (255,0,0), -1)
                cv2.putText(court_img, "HipH", (hh_px[0]+5, hh_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            if 'hip_world_position_ground' in pdata:
                hg = pdata['hip_world_position_ground']
                hg_px = world_to_court(hg[0], hg[1])
                cv2.circle(court_img, hg_px, 5, (0,255,0), -1)
                cv2.putText(court_img, "HipG", (hg_px[0]+5, hg_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Ankle points on overhead
            if 'left_ankle_world' in pdata:
                la = pdata['left_ankle_world']
                la_px = world_to_court(la[0], la[1])
                cv2.circle(court_img, la_px, 5, (0,255,255), -1)
                cv2.putText(court_img, "LAnk", (la_px[0]+5, la_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            if 'right_ankle_world' in pdata:
                ra = pdata['right_ankle_world']
                ra_px = world_to_court(ra[0], ra[1])
                cv2.circle(court_img, ra_px, 5, (255,255,0), -1)
                cv2.putText(court_img, "RAnk", (ra_px[0]+5, ra_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            if 'ankle_midpoint_world' in pdata:
                am = pdata['ankle_midpoint_world']
                am_px = world_to_court(am[0], am[1])
                cv2.circle(court_img, am_px, 5, (255,0,255), -1)
                cv2.putText(court_img, "AnkM", (am_px[0]+5, am_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

            # On original frame, draw joints as detected
            j = pdata['joints']
            # Draw hip height and ground
            if 'hip_world_position_hip_plane' in pdata:
                hh = pdata['hip_world_position_hip_plane']
                hh_px = world_to_court(hh[0], hh[1])
                cv2.circle(court_img, hh_px, 5, (255,0,0), -1)
                cv2.putText(court_img, "HipH", (hh_px[0]+5, hh_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            if 'hip_world_position_ground' in pdata:
                hg = pdata['hip_world_position_ground']
                hg_px = world_to_court(hg[0], hg[1])
                cv2.circle(court_img, hg_px, 5, (0,255,0), -1)
                cv2.putText(court_img, "HipG", (hg_px[0]+5, hg_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            if 'left_ankle' in j:
                la_pt = j['left_ankle'].astype(int)
                cv2.circle(frame, (la_pt[0], la_pt[1]), 5, (0,255,255), -1)
                cv2.putText(frame, "LAnk", (la_pt[0]+5, la_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            if 'right_ankle' in j:
                ra_pt = j['right_ankle'].astype(int)
                cv2.circle(frame, (ra_pt[0], ra_pt[1]), 5, (255,255,0), -1)
                cv2.putText(frame, "RAnk", (ra_pt[0]+5, ra_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            if 'ankle_midpoint' in j:
                am_pt = j['ankle_midpoint'].astype(int)
                cv2.circle(frame, (am_pt[0], am_pt[1]), 5, (255,0,255), -1)
                cv2.putText(frame, "AnkM", (am_pt[0]+5, am_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        combined[:h, :w] = frame
        combined[:court_img.shape[0], w:] = court_img
        output_video.write(combined)

    output_video.release()
    cap.release()
    logging.info(f"Visualization saved to {output_video_path}")

if __name__ == "__main__":
    main()