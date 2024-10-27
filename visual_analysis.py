import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def moving_average(data, window_size=3):
    """
    Applies a moving average to smooth the data.

    Args:
        data (np.ndarray or pd.Series): The data to be smoothed.
        window_size (int): The size of the moving average window.

    Returns:
        np.ndarray: The smoothed data.
    """
    return pd.Series(data).rolling(window=window_size, center=True).mean().values

def detect_jumps(df, base_threshold=0.5, min_jump_duration=3, max_jump_duration=10, max_x_displacement=0.5, window_size=5):
    """
    Detects jumps in player movements based on Y-coordinate changes and vertical velocity.

    Args:
        df (pd.DataFrame): DataFrame containing player positions.
        base_threshold (float): Minimum height above ground level to consider as a jump.
        min_jump_duration (int): Minimum number of frames a jump should last.
        max_jump_duration (int): Maximum number of frames a jump should last.
        max_x_displacement (float): Maximum allowed X displacement during a jump.
        window_size (int): The window size for moving average smoothing.

    Returns:
        list: A list of detected jumps with frame indices and coordinates.
    """
    jumps = []
    players = df['human_index'].unique()

    # Loop through each player separately
    for player in players:
        player_data = df[df['human_index'] == player]
        frames = player_data['frame_index'].values
        y_coords = player_data['world_Y'].values
        x_coords = player_data['world_X'].values

        if len(frames) < window_size:
            continue  # Skip if there's not enough data

        # Apply moving average to smooth the data
        smoothed_y = moving_average(y_coords, window_size=window_size)
        smoothed_x = moving_average(x_coords, window_size=window_size)
        smoothed_frames = frames

        # Replace NaN values resulting from smoothing
        smoothed_y = np.nan_to_num(smoothed_y, nan=np.nanmean(smoothed_y))
        smoothed_x = np.nan_to_num(smoothed_x, nan=np.nanmean(smoothed_x))

        # Calculate ground level as the 5th percentile
        ground_level = np.percentile(smoothed_y, 5)

        # Calculate relative Y (height above ground)
        relative_y = smoothed_y - ground_level

        # Calculate vertical velocity
        velocity_y = np.gradient(smoothed_y, smoothed_frames)

        # Identify potential jump indices where velocity exceeds threshold
        upward_movement = velocity_y > 0.1  # Adjust threshold as needed
        potential_jumps = np.where(upward_movement & (relative_y > base_threshold))[0]

        # Group continuous indices
        if len(potential_jumps) == 0:
            continue

        jump_groups = np.split(potential_jumps, np.where(np.diff(potential_jumps) != 1)[0] + 1)

        for group in jump_groups:
            if len(group) < min_jump_duration or len(group) > max_jump_duration:
                continue  # Skip if duration is not within limits

            start_idx = group[0]
            end_idx = group[-1]

            jump_start_frame = smoothed_frames[start_idx]
            jump_end_frame = smoothed_frames[end_idx]
            jump_duration = jump_end_frame - jump_start_frame + 1

            # Calculate X displacement during the jump
            x_displacement = abs(smoothed_x[end_idx] - smoothed_x[start_idx])
            if x_displacement > max_x_displacement:
                continue  # Skip if X displacement is too large (player is running)

            # Additional check: ensure that the Y reaches a peak and then decreases
            peak_idx = start_idx + np.argmax(relative_y[start_idx:end_idx + 1])
            if peak_idx == start_idx or peak_idx == end_idx:
                continue  # Not a valid peak

            # Ensure that after the peak, the player lands back to ground level
            landing_indices = np.where((smoothed_frames > smoothed_frames[peak_idx]) & (relative_y <= base_threshold))[0]
            if len(landing_indices) == 0:
                continue  # No landing detected
            actual_end_idx = landing_indices[0]
            actual_jump_end_frame = smoothed_frames[actual_end_idx]
            actual_jump_duration = actual_jump_end_frame - jump_start_frame + 1

            if actual_jump_duration > max_jump_duration:
                continue  # Skip if the adjusted duration is too long

            logging.info(f"Player {player}: Jump from frame {jump_start_frame} to {actual_jump_end_frame}, duration {actual_jump_duration} frames, X displacement {x_displacement:.2f}")

            jumps.append({
                'human_index': player,
                'jump_start_frame': jump_start_frame,
                'jump_end_frame': actual_jump_end_frame,
                'jump_start_y': smoothed_y[start_idx],
                'jump_end_y': smoothed_y[actual_end_idx],
                'jump_duration': actual_jump_duration,
                'x_displacement': x_displacement,
                'frames': smoothed_frames[start_idx:actual_end_idx+1],
                'relative_y': relative_y[start_idx:actual_end_idx+1]
            })

    return jumps

def plot_player_movements(df, jumps, player_id, output_file=None):
    """
    Plots the original and smoothed Y-coordinate over time for a specific player and highlights detected jumps.

    Args:
        df (pd.DataFrame): DataFrame containing player positions.
        jumps (list): List of detected jumps.
        player_id (int): ID of the player to plot.
        output_file (str): Path to save the plot image (optional).
    """
    player_data = df[df['human_index'] == player_id]
    frames = player_data['frame_index'].values
    y_coords = player_data['world_Y'].values

    # Apply moving average to smooth the data
    smoothed_y = moving_average(y_coords, window_size=5)
    smoothed_frames = frames

    # Replace NaN values resulting from smoothing
    smoothed_y = np.nan_to_num(smoothed_y, nan=np.nanmean(smoothed_y))

    plt.figure(figsize=(12, 6))

    # Plot original data
    plt.plot(frames, y_coords, label='Original Y-Coordinate', alpha=0.5)

    # Plot smoothed data
    plt.plot(smoothed_frames, smoothed_y, label='Smoothed Y-Coordinate', linewidth=2)

    # Plot detected jumps
    for jump in jumps:
        if jump['human_index'] != player_id:
            continue
        jump_frames = jump['frames']
        jump_relative_y = jump['relative_y'] + np.percentile(smoothed_y, 5)  # Convert back to actual Y-coordinate
        plt.plot(jump_frames, jump_relative_y, color='red', linewidth=2, label='Detected Jump')
        plt.fill_between(jump_frames, jump_relative_y, np.percentile(smoothed_y, 5), color='red', alpha=0.3)

    plt.title(f'Player {player_id} Movements with Detected Jumps')
    plt.xlabel('Frame Index')
    plt.ylabel('Y-Coordinate')
    plt.legend()
    plt.grid(True)

    if output_file:
        plt.savefig(output_file)
        logging.info(f'Plot saved to {output_file}')
    else:
        plt.show()

def main():
    """
    Main function to process the CSV file, detect jumps, and plot the movements.
    """
    parser = argparse.ArgumentParser(description='Visual analysis of player movements and detected jumps.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing player positions.')
    parser.add_argument('--player_id', type=int, default=0, help='ID of the player to analyze.')
    parser.add_argument('--output_plot', type=str, default=None, help='Path to save the plot image.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Minimum height above ground level to consider as a jump.')
    parser.add_argument('--min_jump_duration', type=int, default=3, help='Minimum number of frames a jump should last.')
    parser.add_argument('--max_jump_duration', type=int, default=10, help='Maximum number of frames a jump should last.')
    parser.add_argument('--max_x_displacement', type=float, default=0.5, help='Maximum allowed X displacement during a jump.')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for moving average smoothing.')
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
        jumps = detect_jumps(
            df,
            base_threshold=args.threshold,
            min_jump_duration=args.min_jump_duration,
            max_jump_duration=args.max_jump_duration,
            max_x_displacement=args.max_x_displacement,
            window_size=args.window_size
        )

        # Plot player movements and detected jumps
        plot_player_movements(df, jumps, args.player_id, args.output_plot)

    except Exception as e:
        logging.exception("An error occurred during visual analysis.")
        sys.exit(1)

if __name__ == "__main__":
    main()