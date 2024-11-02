import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

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

# Iterate over each person to remove outliers, detect strong minima, and plot ground truth frames
for pid in person_ids:
    # Filter data for the current person
    person_df = df[df['human_index'] == pid].copy()

    # Sort by 'frame_index' to ensure correct order
    person_df.sort_values('frame_index', inplace=True)

    # ---------------------------
    # Outlier Detection and Removal
    # ---------------------------
    # Compute Q1 and Q3
    Q1 = person_df['world_Y'].quantile(0.25)
    Q3 = person_df['world_Y'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    k = 1.5  # Adjust k as needed
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    # Remove outliers, but keep ground truth frames
    cleaned_df = person_df[
        ((person_df['world_Y'] >= lower_bound) & (person_df['world_Y'] <= upper_bound)) |
        (person_df['frame_index'].isin(ground_truth_frames))
        ].copy()

    # ---------------------------
    # Proceed with Smoothing and Minima Detection
    # ---------------------------

    # Apply Savitzky-Golay filter for smoothing 'world_Y'
    window_length = 21  # You can adjust this as needed
    polyorder = 2       # You can adjust this as needed

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

    # Invert the smoothed 'world_Y' data to find minima as peaks
    inverted_world_Y = -cleaned_df['world_Y_smooth']

    # Find peaks in the inverted data (which correspond to minima in the original data)
    # Remove all constraints to detect all local minima
    peaks, _ = find_peaks(inverted_world_Y)

    # Extract minima values
    minima_frames = cleaned_df['frame_index'].iloc[peaks].tolist()
    minima_values = cleaned_df['world_Y_smooth'].iloc[peaks].tolist()

    # ---------------------------
    # Plotting
    # ---------------------------
    plt.figure(figsize=(12, 7))

    # Plot the cleaned 'world_Y' data
    plt.plot(cleaned_df['frame_index'], cleaned_df['world_Y'], label='Cleaned world_Y', alpha=0.5)

    # Plot the smoothed 'world_Y' data
    plt.plot(cleaned_df['frame_index'], cleaned_df['world_Y_smooth'], label='Smoothed world_Y', linewidth=2)

    # Mark the detected minima
    plt.plot(minima_frames, minima_values, 'go', label='Detected Minima')  # Green circles

    # Highlight the ground truth frames
    ground_truth_data = cleaned_df[cleaned_df['frame_index'].isin(ground_truth_frames)]
    plt.plot(ground_truth_data['frame_index'], ground_truth_data['world_Y_smooth'], 'ro', label='Ground Truth Frames')  # Red circles

    # Annotate the ground truth frames
    for idx, row in ground_truth_data.iterrows():
        plt.annotate(f"Frame {int(row['frame_index'])}", (row['frame_index'], row['world_Y_smooth']),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='red')

    # Annotate the detected minima
    for x, y in zip(minima_frames, minima_values):
        plt.annotate(f"Min at {int(x)}", (x, y),
                     textcoords="offset points", xytext=(0, -15), ha='center', color='green')

    plt.xlabel('Frame Index')
    plt.ylabel('world_Y')
    plt.title(f'Person {pid}: Detected Minima and Ground Truth Frames')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optionally, print out the frames and values of the detected minima
    minima_data = pd.DataFrame({
        'frame_index': minima_frames,
        'world_Y': minima_values
    })
    print(f"\nDetected Minima for Person {pid}:")
    print(minima_data)