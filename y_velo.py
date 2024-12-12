import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import argparse

def remove_outliers(person_df, ground_truth_frames, k=1.5):
    Q1 = person_df['world_Y'].quantile(0.25)
    Q3 = person_df['world_Y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    cleaned_df = person_df[
        ((person_df['world_Y'] >= lower_bound) & (person_df['world_Y'] <= upper_bound)) |
        (person_df['frame_index'].isin(ground_truth_frames))
        ].copy()
    return cleaned_df

def smooth_data(cleaned_df, window_length=0, polyorder=0):
    if len(cleaned_df['world_Y']) < window_length:
        window_length = len(cleaned_df['world_Y'])
        if window_length % 2 == 0:
            window_length -= 1
    if window_length >= 3 and len(cleaned_df['world_Y']) >= window_length:
        cleaned_df['world_Y_smooth'] = savgol_filter(
            cleaned_df['world_Y'],
            window_length=window_length,
            polyorder=polyorder
        )
    else:
        cleaned_df['world_Y_smooth'] = cleaned_df['world_Y']
    return cleaned_df

def detect_strong_minima(cleaned_df, N=5, base_prominence_threshold=None, prominence_threshold_factor=1.0, slope_tolerance=0.5):
    cleaned_df['world_Y_derivative'] = np.gradient(cleaned_df['world_Y_smooth'], cleaned_df['frame_index'])
    inverted_world_Y = -cleaned_df['world_Y_smooth']
    if base_prominence_threshold is None:
        data_range = cleaned_df['world_Y_smooth'].max() - cleaned_df['world_Y_smooth'].min()
        base_prominence_threshold = 0.05 * data_range
    peaks, properties = find_peaks(inverted_world_Y, prominence=base_prominence_threshold)
    prominences = properties['prominences']
    strong_minima_data = []
    for idx, prominence in zip(peaks, prominences):
        frame_idx = cleaned_df['frame_index'].iloc[idx]
        minima_value = cleaned_df['world_Y_smooth'].iloc[idx]
        distance_from_6_7 = abs(minima_value - 6.7)
        adjusted_prominence_threshold = base_prominence_threshold * (1 / (1 + prominence_threshold_factor * distance_from_6_7))
        if prominence < adjusted_prominence_threshold:
            continue
        start_before = max(0, idx - N)
        end_before = idx
        start_after = idx + 1
        end_after = min(len(cleaned_df), idx + N + 1)
        slopes_before = cleaned_df['world_Y_derivative'].iloc[start_before:end_before]
        slopes_after = cleaned_df['world_Y_derivative'].iloc[start_after:end_after]
        avg_slope_before = slopes_before.mean()
        avg_slope_after = slopes_after.mean()
        if avg_slope_before * avg_slope_after < 0:
            abs_avg_slope_before = abs(avg_slope_before)
            abs_avg_slope_after = abs(avg_slope_after)
            if abs_avg_slope_before == 0 or abs_avg_slope_after == 0:
                continue
            ratio = abs_avg_slope_before / abs_avg_slope_after
            if (1 - slope_tolerance) <= ratio <= (1 + slope_tolerance):
                strong_minima_data.append({
                    'frame_index': frame_idx,
                    'world_Y': minima_value,
                    'avg_slope_before': avg_slope_before,
                    'avg_slope_after': avg_slope_after,
                    'prominence': prominence
                })
    minima_df = pd.DataFrame(strong_minima_data)
    return minima_df, cleaned_df

def plot_results(cleaned_df, minima_df, ground_truth_frames, pid, N=5):
    if minima_df.empty:
        print(f"No strong minima detected for tracked_id {pid}.")
        return
    plt.figure(figsize=(12, 7))
    plt.plot(cleaned_df['frame_index'], cleaned_df['world_Y'], label='Cleaned world_Y', alpha=0.5)
    plt.plot(cleaned_df['frame_index'], cleaned_df['world_Y_smooth'], label='Smoothed world_Y', linewidth=2)
    plt.plot(minima_df['frame_index'], minima_df['world_Y'], 'go', label='Detected Strong Minima')
    ground_truth_data = cleaned_df[cleaned_df['frame_index'].isin(ground_truth_frames)]
    plt.plot(ground_truth_data['frame_index'], ground_truth_data['world_Y_smooth'], 'ro', label='Ground Truth Frames')
    for idx, row in ground_truth_data.iterrows():
        plt.annotate(f"Frame {int(row['frame_index'])}", (row['frame_index'], row['world_Y_smooth']),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='red')
    for idx, row in minima_df.iterrows():
        x = row['frame_index']
        y = row['world_Y']
        avg_slope_before = row['avg_slope_before']
        avg_slope_after = row['avg_slope_after']
        prominence = row['prominence']
        plt.annotate(f"Min at {int(x)}\nSlope before: {avg_slope_before:.2f}\nSlope after: {avg_slope_after:.2f}\nProminence: {prominence:.2f}",
                     (x, y), textcoords="offset points", xytext=(0, -60), ha='center', color='green')
    plt.xlabel('Frame Index')
    plt.ylabel('world_Y')
    plt.title(f'tracked_id {pid}: Detected Strong Minima Based on Adaptive Prominence Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(csv_file, show_graph=False, player_id=None):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df['frame_index'] = df['frame_index'].astype(int)
    df['tracked_id'] = df['tracked_id'].astype(int)
    ground_truth_frames = [200, 294, 320, 365, 467]

    # Get frame counts for each tracked_id and sort by frame count
    frame_counts = df.groupby('tracked_id')['frame_index'].nunique().sort_values(ascending=False)
    top_players = frame_counts.index[:2]

    # Use provided player_id if specified, otherwise use the top two tracked_ids
    person_ids = [player_id] if player_id is not None else top_players

    for pid in person_ids:
        person_df = df[df['tracked_id'] == pid].copy()
        person_df.sort_values('frame_index', inplace=True)
        num_person_frames = person_df['frame_index'].nunique()
        print(f"\nProcessing tracked_id {pid} with {num_person_frames} frames.")
        cleaned_df = remove_outliers(person_df, ground_truth_frames, k=1.5)
        cleaned_df = smooth_data(cleaned_df, window_length=25, polyorder=2)
        data_range = cleaned_df['world_Y_smooth'].max() - cleaned_df['world_Y_smooth'].min()
        base_prominence_threshold = 0.05 * data_range
        prominence_threshold_factor = 1.0
        slope_tolerance = 0.5
        N = 5
        minima_df, cleaned_df = detect_strong_minima(
            cleaned_df,
            N=N,
            base_prominence_threshold=base_prominence_threshold,
            prominence_threshold_factor=prominence_threshold_factor,
            slope_tolerance=slope_tolerance
        )
        if show_graph:
            plot_results(cleaned_df, minima_df, ground_truth_frames, pid, N)
        print(f"\nDetected Strong Minima for tracked_id {pid}:")
        print(minima_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and detect minima in world_Y data.")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("--show_graph", action="store_true", help="Whether to show graphs.")
    parser.add_argument("--player_id", type=int, help="Specify the player ID to process. Process top 2 tracked_ids if not specified.")
    args = parser.parse_args()
    main(args.csv_file, show_graph=args.show_graph, player_id=args.player_id)