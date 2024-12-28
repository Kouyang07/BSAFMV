import os
import sys
import cv2
import csv
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------
# 1) Helper functions
# -------------------
def define_world_points():
    """
    3D corner points of the badminton court (same as your main script).
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
    Reads the 2D corner points from a CSV that has 'Point','X','Y' columns.
    Must contain P1, P2, P3, P4 rows.
    """
    df = pd.read_csv(court_csv)
    corner_points = df[df['Point'].isin(['P1', 'P2', 'P3', 'P4'])]
    image_points = corner_points[['X', 'Y']].values.astype(np.float32)
    return image_points

def calibrate_camera(world_points, image_points, camera_matrix):
    """
    Perform camera pose estimation (PnP). Returns rvec, tvec, dist_coeffs
    """
    dist_coeffs = np.zeros((4,1))
    ret, rvec, tvec = cv2.solvePnP(
        world_points,
        image_points,
        camera_matrix,
        dist_coeffs
    )
    if not ret:
        logging.error("Camera pose estimation failed.")
        sys.exit(1)
    return rvec, tvec, dist_coeffs

def draw_court(court_image, court_scale):
    """
    Draw a simplified badminton court on the overhead image (just lines).
    """
    court_length_m = 13.4
    court_width_m = 6.1
    margin_m = 2.0
    left = int(margin_m*court_scale)
    right = int((margin_m+court_width_m)*court_scale)
    top = int(margin_m*court_scale)
    bottom = int((margin_m+court_length_m)*court_scale)
    cv2.rectangle(court_image, (left, top), (right, bottom), (255,255,255), 2)
    # Draw net line at halfway (6.7m)
    net_y = int((margin_m + court_length_m/2)*court_scale)
    cv2.line(court_image, (left, net_y), (right, net_y), (255,255,255), 2)

def world_to_court(X, Y, court_scale, court_length_m=13.4, margin_m=2.0):
    """
    Convert real-world (X, Y) to overhead court coordinates in pixels.
    Y grows downward in the overhead image, so invert Y accordingly.
    """
    px = int((X + margin_m) * court_scale)
    py = int((court_length_m + margin_m - Y) * court_scale)
    return (px, py)

# ---------------------------
# 2) Visualization main logic
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Visualize corrected Y-values for multiple players (each in separate CSV).')
    parser.add_argument('video_path', type=str,
                        help='Path to the original video (e.g. input_video.mp4).')
    parser.add_argument('--positions_csv', type=str, default=None,
                        help='Path to the original positions CSV (containing all players). '
                             'If not provided, auto-locate e.g. result/<video_name>_positions.csv.')
    parser.add_argument('--court_csv', type=str, default=None,
                        help='Path to the <video_name>_court.csv file. If not provided, auto-locate.')
    parser.add_argument('--corrected_csvs', nargs='+', default=None,
                        help='One or more corrected CSV paths (one CSV per player). '
                             'If not provided, attempt auto-locate for 2 players: tracked_id_0 and tracked_id_1.')
    parser.add_argument('--output_video', type=str, default=None,
                        help='Path to output video with side-by-side view. If not provided, auto-locate.')
    args = parser.parse_args()

    # 2.1) Derive the base name from the video
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]

    # 2.2) Auto-locate positions CSV if not given
    if args.positions_csv:
        positions_csv = args.positions_csv
    else:
        # e.g. result/<video_name>_positions.csv
        positions_csv = os.path.join('result', f"{base_name}_positions.csv")

    # 2.3) Auto-locate corrected CSVs if not given
    if args.corrected_csvs is None or len(args.corrected_csvs) == 0:
        # We'll guess the user has 2 players: tracked_id_0, tracked_id_1
        # Adjust or add more if you have more players
        guessed_csv0 = os.path.join('result', f"{base_name}_positions_tracked_id_0_corrected_positions.csv")
        guessed_csv1 = os.path.join('result', f"{base_name}_positions_tracked_id_1_corrected_positions.csv")
        corrected_csvs = [guessed_csv0, guessed_csv1]
    else:
        corrected_csvs = args.corrected_csvs

    # 2.4) Auto-locate court CSV if not given
    if args.court_csv:
        court_csv = args.court_csv
    else:
        court_csv = os.path.join('result', f"{base_name}_court.csv")

    # 2.5) Auto-locate output video if not given
    if args.output_video:
        output_video = args.output_video
    else:
        # e.g. result/<video_name>_corrected_overlay.mp4
        output_video = os.path.join('result', f"{base_name}_corrected_overlay.mp4")

    # Print out resolved paths (for debugging/logging)
    logging.info("Resolved paths:")
    logging.info(f"  video_path     = {args.video_path}")
    logging.info(f"  positions_csv  = {positions_csv}")
    logging.info(f"  corrected_csvs = {corrected_csvs}")
    logging.info(f"  court_csv      = {court_csv}")
    logging.info(f"  output_video   = {output_video}")

    # 2.6) Check existence of required files
    if not os.path.exists(positions_csv):
        logging.error(f"Positions CSV not found: {positions_csv}")
        sys.exit(1)
    missing_corrected = [c for c in corrected_csvs if not os.path.exists(c)]
    if missing_corrected:
        logging.error("Some corrected CSVs not found:\n" + "\n".join(missing_corrected))
        sys.exit(1)
    if not os.path.exists(court_csv):
        logging.error(f"Court CSV not found: {court_csv}")
        sys.exit(1)

    # Ensure output directory
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    # 2.7) Load the single original CSV
    df_orig = pd.read_csv(positions_csv)
    if not {'frame_index','tracked_id','world_X','world_Y'}.issubset(df_orig.columns):
        logging.error("Original CSV must contain frame_index, tracked_id, world_X, world_Y columns.")
        sys.exit(1)

    # Build a dictionary for the original data:
    #   orig_dict[tid][frame_idx] = (world_X, world_Y)
    orig_dict = {}
    for row in df_orig.itertuples():
        fidx = row.frame_index
        tid  = row.tracked_id
        wX   = row.world_X
        wY   = row.world_Y
        if tid not in orig_dict:
            orig_dict[tid] = {}
        orig_dict[tid][fidx] = (wX, wY)

    # 2.8) Load each corrected CSV separately (one CSV per tracked_id)
    # Build corr_dict[tid][frame_idx] = (world_X, corrected_world_Y)
    corr_dict = {}

    for csv_path in corrected_csvs:
        dfc = pd.read_csv(csv_path)
        # We expect columns: frame_index, tracked_id, world_X, corrected_world_Y, ...
        required_cols = {'frame_index','tracked_id','world_X','corrected_world_Y'}
        if not required_cols.issubset(dfc.columns):
            logging.error(f"{csv_path} missing one of required columns: {required_cols}")
            sys.exit(1)

        # Check if there's exactly one tid in this CSV
        tids_in_file = dfc['tracked_id'].unique()
        if len(tids_in_file) != 1:
            logging.error(f"{csv_path} must contain exactly one tracked_id, but found: {tids_in_file}")
            sys.exit(1)

        this_tid = tids_in_file[0]
        logging.info(f"Reading corrected CSV for TID={this_tid}: {csv_path}")

        # Make sure our dictionary has a slot
        if this_tid not in corr_dict:
            corr_dict[this_tid] = {}

        # Populate it
        for row in dfc.itertuples():
            fidx   = row.frame_index
            wX     = row.world_X
            wYcorr = row.corrected_world_Y
            corr_dict[this_tid][fidx] = (wX, wYcorr)

    # 2.9) Open the video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {args.video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2.10) Approximate camera intrinsics
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # 2.11) Calibrate
    world_pts = define_world_points()
    image_pts = read_court_points(court_csv)
    rvec, tvec, dist_coeffs = calibrate_camera(world_pts, image_pts, camera_matrix)
    # (We don't necessarily use rvec/tvec in this script, but we keep the same structure as your pipeline.)

    # 2.12) Prepare overhead court image template
    court_scale   = 50
    court_length_m= 13.4
    court_width_m = 6.1
    margin_m      = 2.0
    court_img_h   = int((court_length_m + 2*margin_m) * court_scale)
    court_img_w   = int((court_width_m + 2*margin_m) * court_scale)

    court_img_template = np.zeros((court_img_h, court_img_w, 3), dtype=np.uint8)
    draw_court(court_img_template, court_scale)

    out_w = w + court_img_w
    out_h = max(h, court_img_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))
    if not out_video.isOpened():
        logging.error(f"Cannot open video writer: {output_video}")
        sys.exit(1)

    # For color-coding different tracked IDs
    # (Add more colors if you have more than ~7 players)
    colors = [
        (0,255,0),   # green
        (0,0,255),   # red
        (255,0,255), # magenta
        (255,255,0), # cyan
        (0,255,255), # yellow
        (128,128,255),
        (255,128,128)
    ]
    def get_color(tid):
        return colors[tid % len(colors)]

    logging.info("Starting to generate visualization...")

    # 2.13) Render each frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f_idx in tqdm(range(frame_count), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        overhead = court_img_template.copy()
        # Label the frame
        cv2.putText(frame, f"Frame: {f_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # We'll combine TIDs from both the original dictionary and the corrected dictionary
        all_tids = set(orig_dict.keys()).union(set(corr_dict.keys()))

        for tid in sorted(all_tids):
            color = get_color(tid)

            # Draw original positions (if available for this frame)
            if tid in orig_dict and f_idx in orig_dict[tid]:
                (wX, wY) = orig_dict[tid][f_idx]
                px, py   = world_to_court(wX, wY, court_scale)
                cv2.circle(overhead, (px, py), 5, color, -1)
                cv2.putText(overhead, f"ID{tid}-O", (px+5, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw corrected positions (if available for this frame)
            if tid in corr_dict and f_idx in corr_dict[tid]:
                (wXcorr, wYcorr) = corr_dict[tid][f_idx]
                px_corr, py_corr = world_to_court(wXcorr, wYcorr, court_scale)
                cv2.circle(overhead, (px_corr, py_corr), 8, (255,255,255), 2)  # white ring
                cv2.circle(overhead, (px_corr, py_corr), 4, color, -1)        # inner color
                cv2.putText(overhead, f"ID{tid}-C", (px_corr+5, py_corr-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Side-by-side output
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        combined[:h, :w] = frame
        combined[:overhead.shape[0], w:] = overhead

        out_video.write(combined)

    out_video.release()
    cap.release()
    logging.info(f"Done! Visualization saved at {output_video}")

if __name__ == "__main__":
    main()