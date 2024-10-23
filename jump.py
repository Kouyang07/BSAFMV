"python jump.py result/test_position.csv --threshold 3 --min_jump_duration 5"

import sys

import pandas as pd
import numpy as np
import logging
import argparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def moving_average(data, window_size=3):
    """
    Applies a moving average to smooth the data.

    Args:
        data (np.ndarray): The data to be smoothed.
        window_size (int): The size of the moving average window.

    Returns:
        np.ndarray: The smoothed data.
    """
    if len(data) < window_size:
        return data  # Return original data if it's smaller than the window size
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def detect_jumps(csv_file, base_threshold=5, velocity_threshold=2.5, min_jump_duration=5, window_size=3):
    """
    Detects jumps in player movements based on Y-coordinate changes and velocity.

    Args:
        csv_file (str): Path to the CSV file with player positions.
        base_threshold (float): Minimum Y-difference to consider a jump.
        velocity_threshold (float): Minimum velocity to indicate a jump.
        min_jump_duration (int): Minimum number of frames a jump should last.
        window_size (int): The window size for moving average smoothing.

    Returns:
        list: A list of detected jumps with frame indices and Y-coordinates.
    """
    try:
        df = pd.read_csv(csv_file)

        # Ensure the dataframe is sorted by frame and human index
        df = df.sort_values(by=['frame_index', 'human_index'])

        jumps = []
        players = df['human_index'].unique()

        # Loop through each player separately
        for player in players:
            player_data = df[df['human_index'] == player]
            y_coords = player_data['world_Y'].values
            frames = player_data['frame_index'].values

            # Apply moving average to smooth the Y-values
            smoothed_y = moving_average(y_coords, window_size=window_size)

            # Adjust frames length to match the smoothed Y-values length
            smoothed_frames = frames[len(frames) - len(smoothed_y):]

            in_jump = False
            jump_start_frame = None
            jump_start_y = None

            previous_y = smoothed_y[0]

            for i in range(1, len(smoothed_y)):
                current_y = smoothed_y[i]
                frame = smoothed_frames[i]

                # Calculate velocity (change in Y per frame)
                velocity_y = current_y - previous_y

                # Detect jump start (sudden upward movement with sufficient velocity)
                if velocity_y > velocity_threshold and current_y > previous_y + base_threshold and not in_jump:
                    in_jump = True
                    jump_start_frame = frame
                    jump_start_y = current_y
                    logging.info(f"Player {player}: Jump started at frame {jump_start_frame} with Y={jump_start_y:.2f}")

                # Detect jump end (when Y-coordinate returns to a near-ground value with downward velocity)
                elif velocity_y < -velocity_threshold and in_jump:
                    in_jump = False
                    jump_end_frame = frame
                    jump_end_y = current_y
                    jump_duration = jump_end_frame - jump_start_frame

                    if jump_duration >= min_jump_duration:
                        logging.info(f"Player {player}: Jump ended at frame {jump_end_frame} with Y={jump_end_y:.2f}")
                        jumps.append({
                            'human_index': player,
                            'jump_start_frame': jump_start_frame,
                            'jump_end_frame': jump_end_frame,
                            'jump_start_y': jump_start_y,
                            'jump_end_y': jump_end_y,
                            'jump_duration': jump_duration
                        })

                previous_y = current_y

        return jumps
    except FileNotFoundError as e:
        logging.error(f"File {csv_file} not found.")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"File {csv_file} is empty.")
        raise e


def save_jumps_to_csv(jumps, output_file):
    """
    Saves the detected jumps to a CSV file.

    Args:
        jumps (list): List of detected jumps.
        output_file (str): Path to the output CSV file.
    """
    if len(jumps) == 0:
        logging.warning("No jumps detected.")
        return

    # Convert list of jumps to a DataFrame
    df_jumps = pd.DataFrame(jumps)

    # Save to CSV
    df_jumps.to_csv(output_file, index=False)
    logging.info(f"Jumps saved to {output_file}")


def main():
    """
    Main function to process the CSV file and detect jumps.
    """
    parser = argparse.ArgumentParser(description='Detect jumps in player positions from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing player positions.')
    parser.add_argument('--output_csv', type=str, default='detected_jumps.csv',
                        help='Path to output CSV file for detected jumps.')
    parser.add_argument('--threshold', type=float, default=5, help='Y-difference threshold to consider a jump.')
    parser.add_argument('--velocity_threshold', type=float, default=2.5,
                        help='Velocity threshold to indicate jump movement.')
    parser.add_argument('--min_jump_duration', type=int, default=5, help='Minimum number of frames a jump should last.')
    parser.add_argument('--window_size', type=int, default=3, help='Window size for moving average smoothing.')
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Detect jumps
        jumps = detect_jumps(args.csv_file, base_threshold=args.threshold, velocity_threshold=args.velocity_threshold,
                             min_jump_duration=args.min_jump_duration, window_size=args.window_size)

        # Save detected jumps to CSV
        save_jumps_to_csv(jumps, args.output_csv)

    except Exception as e:
        logging.exception("An error occurred while detecting jumps.")
        sys.exit(1)


if __name__ == "__main__":
    main()
