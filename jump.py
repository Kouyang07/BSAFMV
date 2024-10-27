import sys
import pandas as pd
import numpy as np
import logging
import argparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_spike_jumps(df, spike_threshold=5, min_jump_duration=2, max_jump_duration=10, max_x_displacement=0.3):
    """
    Detects jumps based on rapid upward and downward spikes in Y-coordinate.

    Args:
        df (pd.DataFrame): DataFrame containing player positions.
        spike_threshold (float): Minimum Y increase followed by a decrease to detect a spike.
        min_jump_duration (int): Minimum frames a jump should last.
        max_jump_duration (int): Maximum frames a jump should last.
        max_x_displacement (float): Maximum allowed X displacement during a jump.

    Returns:
        list: A list of detected jumps with frame indices and coordinates.
    """
    jumps = []
    players = df['human_index'].unique()

    for player in players:
        player_data = df[df['human_index'] == player].copy()
        player_data.sort_values('frame_index', inplace=True)
        frames = player_data['frame_index'].values
        y_coords = player_data['world_Y'].values
        x_coords = player_data['world_X'].values

        # Calculate the velocity (difference in Y-coordinate between frames)
        delta_y = np.diff(y_coords)
        delta_frames = np.diff(frames)
        velocity_y = delta_y / delta_frames

        # Set up variables for detecting jumps
        in_jump = False
        jump_start_idx = None
        jump_peak_y = None

        for i in range(1, len(velocity_y)):
            current_y = y_coords[i]
            previous_y = y_coords[i - 1]

            # Detect the start of a jump by a large increase in Y
            if velocity_y[i - 1] > spike_threshold and not in_jump:
                in_jump = True
                jump_start_idx = i
                jump_peak_y = current_y
                logging.info(f"Player {player}: Potential jump start at frame {frames[i]}, Y = {current_y:.2f}")

            # Detect the end of a jump by a large decrease in Y back to near the starting level
            elif in_jump and (previous_y - current_y) > spike_threshold:
                in_jump = False
                jump_end_idx = i
                jump_duration = frames[jump_end_idx] - frames[jump_start_idx]

                # Check that the jump duration is within acceptable range
                if jump_duration < min_jump_duration or jump_duration > max_jump_duration:
                    continue

                # Check that X displacement is minimal
                x_displacement = abs(x_coords[jump_end_idx] - x_coords[jump_start_idx])
                if x_displacement > max_x_displacement:
                    continue

                # Log and store jump
                logging.info(f"Player {player}: Detected jump from frame {frames[jump_start_idx]} to {frames[jump_end_idx]}, duration {jump_duration} frames, X displacement {x_displacement:.2f}")
                jumps.append({
                    'human_index': player,
                    'jump_start_frame': frames[jump_start_idx],
                    'jump_end_frame': frames[jump_end_idx],
                    'jump_start_y': y_coords[jump_start_idx],
                    'jump_end_y': y_coords[jump_end_idx],
                    'jump_peak_y': jump_peak_y,
                    'jump_duration': jump_duration,
                    'x_displacement': x_displacement
                })

        if not jumps:
            logging.warning("No jumps detected for player {}".format(player))

    return jumps

def main():
    """
    Main function to process the CSV file and detect jumps based on spikes.
    """
    parser = argparse.ArgumentParser(description='Detect jumps in player positions from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing player positions.')
    parser.add_argument('--spike_threshold', type=float, default=5, help='Threshold for detecting a Y-coordinate spike as a jump.')
    parser.add_argument('--min_jump_duration', type=int, default=2, help='Minimum number of frames a jump should last.')
    parser.add_argument('--max_jump_duration', type=int, default=10, help='Maximum number of frames a jump should last.')
    parser.add_argument('--max_x_displacement', type=float, default=0.3, help='Maximum allowed X displacement during a jump.')
    args = parser.parse_args()

    try:
        # Read CSV file
        df = pd.read_csv(args.csv_file)

        # Check for required columns
        required_columns = ['frame_index', 'human_index', 'world_Y', 'world_X']
        if not all(col in df.columns for col in required_columns):
            missing_cols = set(required_columns) - set(df.columns)
            logging.error(f"Missing columns in the CSV file: {missing_cols}")
            sys.exit(1)

        # Detect jumps
        jumps = detect_spike_jumps(
            df,
            spike_threshold=args.spike_threshold,
            min_jump_duration=args.min_jump_duration,
            max_jump_duration=args.max_jump_duration,
            max_x_displacement=args.max_x_displacement
        )

        if not jumps:
            logging.warning("No jumps detected.")

    except Exception as e:
        logging.exception("An error occurred during jump detection.")
        sys.exit(1)

if __name__ == "__main__":
    main()