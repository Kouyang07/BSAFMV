import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

def remove_outliers(person_df, ground_truth_frames, k=1.5):
    """
    Remove outliers from 'world_Y' based on IQR, but preserve ground truth frames.
    """
    # Compute Q1 and Q3
    Q1 = person_df['world_Y'].quantile(0.25)
    Q3 = person_df['world_Y'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    # Remove outliers, but keep ground truth frames
    cleaned_df = person_df[
        ((person_df['world_Y'] >= lower_bound) & (person_df['world_Y'] <= upper_bound)) |
        (person_df['frame_index'].isin(ground_truth_frames))
        ].copy()
    return cleaned_df

def smooth_data(cleaned_df, window_length=4, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth 'world_Y' data.
    """
    # Ensure window_length is appropriate
    if len(cleaned_df['world_Y']) < window_length:
        # Reduce window_length if data is too short
        window_length = len(cleaned_df['world_Y'])
        if window_length % 2 == 0:
            window_length -= 1  # Make it odd

    if window_length >= 3 and len(cleaned_df['world_Y']) >= window_length:
        # Apply smoothing
        cleaned_df['world_Y_smooth'] = savgol_filter(
            cleaned_df['world_Y'],
            window_length=window_length,
            polyorder=polyorder
        )
    else:
        # If not enough data points, skip smoothing
        cleaned_df['world_Y_smooth'] = cleaned_df['world_Y']
    return cleaned_df

def detect_strong_minima(cleaned_df, N=5, base_prominence_threshold=None, prominence_threshold_factor=1.0, slope_tolerance=0.5):
    """
    Detect strong minima with an adaptive prominence threshold based on distance from 6.7.
    """
    # Compute the first derivative of the smoothed 'world_Y'
    cleaned_df['world_Y_derivative'] = np.gradient(cleaned_df['world_Y_smooth'], cleaned_df['frame_index'])

    # Invert the smoothed 'world_Y' data to find minima as peaks
    inverted_world_Y = -cleaned_df['world_Y_smooth']

    # Use a low base prominence threshold to detect more minima
    if base_prominence_threshold is None:
        data_range = cleaned_df['world_Y_smooth'].max() - cleaned_df['world_Y_smooth'].min()
        base_prominence_threshold = 0.05 * data_range  # Adjust as needed

    # Find all minima in the inverted data, get properties including prominence
    peaks, properties = find_peaks(inverted_world_Y, prominence=base_prominence_threshold)

    prominences = properties['prominences']

    # Initialize list to store strong minima data
    strong_minima_data = []

    # Loop over each detected minima
    for idx, prominence in zip(peaks, prominences):
        # Get frame index and value of minima
        frame_idx = cleaned_df['frame_index'].iloc[idx]
        minima_value = cleaned_df['world_Y_smooth'].iloc[idx]

        # Compute adjusted prominence threshold based on distance from 6.7
        distance_from_6_7 = abs(minima_value - 6.7)
        # For minima near 6.7, require higher prominence; for minima far from 6.7, allow lower prominence
        adjusted_prominence_threshold = base_prominence_threshold * (1 / (1 + prominence_threshold_factor * distance_from_6_7))

        # Check if the prominence is above the adjusted threshold
        if prominence < adjusted_prominence_threshold:
            continue  # Skip this minima

        # Define the window for slopes before and after the minima
        start_before = max(0, idx - N)
        end_before = idx  # up to but not including idx
        start_after = idx + 1  # start after the minima
        end_after = min(len(cleaned_df), idx + N + 1)

        # Get the derivatives before and after
        slopes_before = cleaned_df['world_Y_derivative'].iloc[start_before:end_before]
        slopes_after = cleaned_df['world_Y_derivative'].iloc[start_after:end_after]

        # Compute average slopes
        avg_slope_before = slopes_before.mean()
        avg_slope_after = slopes_after.mean()

        # Check if the slopes have opposite signs
        if avg_slope_before * avg_slope_after < 0:
            # Compute the ratio of the absolute slopes
            abs_avg_slope_before = abs(avg_slope_before)
            abs_avg_slope_after = abs(avg_slope_after)

            # Avoid division by zero
            if abs_avg_slope_before == 0 or abs_avg_slope_after == 0:
                continue  # Skip this minima

            ratio = abs_avg_slope_before / abs_avg_slope_after

            # Check if the ratio is within the relaxed tolerance
            if (1 - slope_tolerance) <= ratio <= (1 + slope_tolerance):
                # This is a strong minima
                strong_minima_data.append({
                    'frame_index': frame_idx,
                    'world_Y': minima_value,
                    'avg_slope_before': avg_slope_before,
                    'avg_slope_after': avg_slope_after,
                    'prominence': prominence
                })

    # Convert the list to a DataFrame
    minima_df = pd.DataFrame(strong_minima_data)

    return minima_df, cleaned_df

def plot_results(cleaned_df, minima_df, ground_truth_frames, pid, N=5):
    """
    Plot the cleaned, smoothed data, detected strong minima, and ground truth frames.
    """
    plt.figure(figsize=(12, 7))

    # Plot the cleaned 'world_Y' data
    plt.plot(cleaned_df['frame_index'], cleaned_df['world_Y'], label='Cleaned world_Y', alpha=0.5)

    # Plot the smoothed 'world_Y' data
    plt.plot(cleaned_df['frame_index'], cleaned_df['world_Y_smooth'], label='Smoothed world_Y', linewidth=2)

    # Mark the detected strong minima
    plt.plot(minima_df['frame_index'], minima_df['world_Y'], 'go', label='Detected Strong Minima')  # Green circles

    # Highlight the ground truth frames
    ground_truth_data = cleaned_df[cleaned_df['frame_index'].isin(ground_truth_frames)]
    plt.plot(ground_truth_data['frame_index'], ground_truth_data['world_Y_smooth'], 'ro', label='Ground Truth Frames')  # Red circles

    # Annotate the ground truth frames
    for idx, row in ground_truth_data.iterrows():
        plt.annotate(f"Frame {int(row['frame_index'])}", (row['frame_index'], row['world_Y_smooth']),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='red')

    # Annotate the detected strong minima with their average slopes and prominence
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
    plt.title(f'Person {pid}: Detected Strong Minima Based on Adaptive Prominence Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Read the CSV file located at 'result/test3_position.csv'
    df = pd.read_csv('result/test3_position.csv')

    # Ensure there are no leading/trailing whitespaces in column names
    df.columns = df.columns.str.strip()

    # Verify the column names
    print("Column Names:", df.columns.tolist())

    # Ensure 'frame_index' and 'human_index' are integers
    df['frame_index'] = df['frame_index'].astype(int)
    df['human_index'] = df['human_index'].astype(int)

    # Ground truth frames to be plotted and preserved
    ground_truth_frames = [200, 294, 320, 365, 467]

    # Get a list of unique person IDs
    person_ids = df['human_index'].unique()

    # Compute the total number of frames
    total_frames = df['frame_index'].nunique()
    print(f"Total number of frames: {total_frames}")

    # Iterate over each person to remove outliers, detect strong minima, and plot ground truth frames
    for pid in person_ids:
        # Filter data for the current person
        person_df = df[df['human_index'] == pid].copy()

        # Sort by 'frame_index' to ensure correct order
        person_df.sort_values('frame_index', inplace=True)

        # Compute the number of frames for this person
        num_person_frames = person_df['frame_index'].nunique()

        # Check if the player has less than 80% of the total frames
        if num_person_frames < 0.8 * total_frames:
            print(f"Person {pid} has {num_person_frames} frames which is less than 80% of total frames ({total_frames}). Skipping this person.")
            continue

        # Outlier Detection and Removal
        cleaned_df = remove_outliers(person_df, ground_truth_frames, k=1.5)

        # Proceed with Smoothing
        cleaned_df = smooth_data(cleaned_df, window_length=25, polyorder=2)

        # Compute the data range to set base prominence threshold
        data_range = cleaned_df['world_Y_smooth'].max() - cleaned_df['world_Y_smooth'].min()
        base_prominence_threshold = 0.05 * data_range  # Lowered base prominence threshold

        # Set prominence threshold factor and slope tolerance
        prominence_threshold_factor = 1.0  # Adjust this factor to control how prominence threshold adapts
        slope_tolerance = 0.5  # Increased tolerance to 50%

        # Detect Strong Minima with adaptive prominence threshold
        N = 5  # Number of frames before and after the minima to consider
        minima_df, cleaned_df = detect_strong_minima(
            cleaned_df,
            N=N,
            base_prominence_threshold=base_prominence_threshold,
            prominence_threshold_factor=prominence_threshold_factor,
            slope_tolerance=slope_tolerance
        )

        # Plotting
        plot_results(cleaned_df, minima_df, ground_truth_frames, pid, N)

        # Optionally, print out the frames and values of the detected strong minima
        print(f"\nDetected Strong Minima for Person {pid}:")
        print(minima_df)

if __name__ == "__main__":
    main()