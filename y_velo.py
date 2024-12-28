import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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


def detect_local_extrema(cleaned_df, distance=5, prominence=0.0):
    y = cleaned_df['world_Y'].values
    x = cleaned_df['frame_index'].values

    # Detect maxima in the original signal
    maxima_indices, _ = find_peaks(y, distance=distance, prominence=prominence)

    # Detect minima by finding peaks in the inverted signal
    inverted_y = -y
    minima_indices, _ = find_peaks(inverted_y, distance=distance, prominence=prominence)

    maxima_df = pd.DataFrame({
        'frame_index': x[maxima_indices],
        'world_Y': y[maxima_indices]
    })
    minima_df = pd.DataFrame({
        'frame_index': x[minima_indices],
        'world_Y': y[minima_indices]
    })

    return maxima_df, minima_df
def detect_jumps(cleaned_df, distance=5, prominence=0.0, percentile_cutoff=80,
                 max_search_frames=50, min_height_diff=0.1):
    """
    Detects jumps in the cleaned data, but ensures that the 'next' maximum is 'strong'.

    :param cleaned_df: A pandas DataFrame with 'frame_index' and 'world_Y' columns.
    :param distance: Minimum distance between peaks (passed to `find_peaks`).
    :param prominence: Minimum prominence (passed to `find_peaks`).
    :param percentile_cutoff: Only keep jumps whose prominence is in the top X percentile.
    :param max_search_frames: How many frames after the minimum to look for a 'strong' max.
    :param min_height_diff: Minimum height difference from the local min to consider a max 'strong'.
    :return: A DataFrame of detected jumps with their prominence.
    """
    maxima_df, minima_df = detect_local_extrema(cleaned_df, distance, prominence)

    maxima_df.sort_values('frame_index', inplace=True)
    minima_df.sort_values('frame_index', inplace=True)

    maxima_frames = maxima_df['frame_index'].tolist()

    jumps_list = []
    for _, row in minima_df.iterrows():
        min_frame = row['frame_index']
        min_value = row['world_Y']

        # 1. Find the previous local max
        prev_candidates = [m for m in maxima_frames if m < min_frame]
        if not prev_candidates:
            continue
        prev_max_frame = max(prev_candidates)
        prev_max_value = maxima_df.loc[maxima_df['frame_index'] == prev_max_frame, 'world_Y'].values[0]

        # 2. Find the next local max candidates (within a search window, for instance)
        next_candidates = [m for m in maxima_frames
                           if m > min_frame and m <= min_frame + max_search_frames]
        if not next_candidates:
            continue

        # 3. Among the candidates, pick the "strong" maximum
        candidate_rows = maxima_df[maxima_df['frame_index'].isin(next_candidates)].copy()
        # Filter out maxima that are too close in height to the min_value
        candidate_rows = candidate_rows[candidate_rows['world_Y'] >= min_value + min_height_diff]
        if candidate_rows.empty:
            # If no 'strong' max found, skip
            continue

        # Pick the candidate with the highest world_Y (i.e., the strongest landing)
        idx_strong = candidate_rows['world_Y'].idxmax()
        next_max_frame = candidate_rows.loc[idx_strong, 'frame_index']
        next_max_value = candidate_rows.loc[idx_strong, 'world_Y']

        # 4. Compute jump prominence
        avg_surrounding_max = (prev_max_value + next_max_value) / 2.0
        jump_prominence = avg_surrounding_max - min_value

        jumps_list.append({
            'frame_index': min_frame,
            'world_Y': min_value,
            'prev_max_frame': prev_max_frame,
            'prev_max_value': prev_max_value,
            'next_max_frame': next_max_frame,
            'next_max_value': next_max_value,
            'prominence': jump_prominence
        })

    jumps_df = pd.DataFrame(jumps_list)
    if jumps_df.empty:
        return jumps_df

    # 5. Only keep jumps above a certain prominence percentile
    cutoff_value = np.percentile(jumps_df['prominence'], percentile_cutoff)
    jumps_df = jumps_df[jumps_df['prominence'] >= cutoff_value].copy()
    return jumps_df


def correct_jumps(cleaned_df, jumps_df):
    corrected_df = cleaned_df.copy()
    corrected_df['corrected_world_Y'] = corrected_df['world_Y']

    jumps_df = jumps_df.sort_values(by='frame_index')

    for _, jump in jumps_df.iterrows():
        prev_frame = jump['prev_max_frame']
        prev_value = jump['prev_max_value']
        next_frame = jump['next_max_frame']
        next_value = jump['next_max_value']

        mask = (corrected_df['frame_index'] >= prev_frame) & (corrected_df['frame_index'] <= next_frame)
        sub_df = corrected_df.loc[mask].copy()

        sub_df['corrected_world_Y'] = np.interp(
            sub_df['frame_index'],
            [prev_frame, next_frame],
            [prev_value, next_value]
        )
        corrected_df.loc[mask, 'corrected_world_Y'] = sub_df['corrected_world_Y']

    return corrected_df


def save_corrected_positions(base_file_name, corrected_df, tracked_id):
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{base_file_name}_tracked_id_{tracked_id}_corrected_positions.csv")
    corrected_df.to_csv(output_file, index=False)
    print(f"Corrected positions saved to {output_file}")


def main(csv_file, show_graph=False, player_id=None, percentile_cutoff=80):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df['frame_index'] = df['frame_index'].astype(int)
    df['tracked_id'] = df['tracked_id'].astype(int)

    ground_truth_frames = [200, 294, 320, 365, 467]

    frame_counts = df.groupby('tracked_id')['frame_index'].nunique().sort_values(ascending=False)
    top_players = frame_counts.index[:2]

    person_ids = [player_id] if player_id is not None else top_players

    base_file_name = os.path.splitext(os.path.basename(csv_file))[0]

    for pid in person_ids:
        person_df = df[df['tracked_id'] == pid].copy()
        person_df.sort_values('frame_index', inplace=True)
        num_person_frames = person_df['frame_index'].nunique()
        print(f"\nProcessing tracked_id {pid} with {num_person_frames} frames.")

        cleaned_df = remove_outliers(person_df, ground_truth_frames, k=1.5)

        jumps_df = detect_jumps(
            cleaned_df,
            distance=5,
            prominence=0.0,
            percentile_cutoff=percentile_cutoff
        )
        print(f"Detected jumps for tracked_id {pid}:\n", jumps_df)

        corrected_df = correct_jumps(cleaned_df, jumps_df)

        if show_graph:
            plt.figure(figsize=(12, 6))
            plt.plot(cleaned_df['frame_index'], cleaned_df['world_Y'], label='Original', alpha=0.5)
            plt.plot(corrected_df['frame_index'], corrected_df['corrected_world_Y'], label='Corrected', linewidth=2)

            if not jumps_df.empty:
                plt.scatter(jumps_df['frame_index'], jumps_df['world_Y'], color='red', label='Detected Jump Minima')

            plt.title(f"Tracked_id {pid} - Jump Correction")
            plt.xlabel('Frame Index')
            plt.ylabel('world_Y')
            plt.legend()
            plt.grid(True)
            plt.show()

        save_corrected_positions(base_file_name, corrected_df, pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and correct jumps in world_Y data.")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("--show_graph", action="store_true", help="Whether to show graphs.")
    parser.add_argument("--player_id", type=int,
                        help="Specify a single player ID to process; else process top 2 tracked_ids.")
    parser.add_argument("--percentile_cutoff", type=float, default=85.0,
                        help="Minima must be >= this percentile of jump prominence to be corrected (default=80).")
    args = parser.parse_args()

    main(
        args.csv_file,
        show_graph=args.show_graph,
        player_id=args.player_id,
        percentile_cutoff=args.percentile_cutoff
    )