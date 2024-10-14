import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Read the data from the CSV file
df = pd.read_csv('result/test_shuttle.csv')

# Initialize a DataFrame to store hit frames
hits_df = pd.DataFrame(columns=['Frame'])

# Calculate differences in positions
df['dX'] = df['X'].diff()
df['dY'] = df['Y'].diff()

# Calculate angle of movement and its change
df['Angle'] = np.arctan2(df['dY'], df['dX'])
df['Angle_Change'] = df['Angle'].diff().abs()
df['Angle_Change'].fillna(0, inplace=True)

# Calculate speed and its change
df['Speed'] = np.sqrt(df['dX']**2 + df['dY']**2)
df['Speed_Change'] = df['Speed'].diff().abs()
df['Speed_Change'].fillna(0, inplace=True)

# Set detection thresholds
direction_threshold = np.radians(80)  # Adjusted for your data
speed_change_threshold = df['Speed_Change'].mean() + 2 * df['Speed_Change'].std()

# Method 1: Visibility Change
df['Visibility_Change'] = df['Visibility'].diff()
hits_visibility = df[df['Visibility_Change'] == 1]['Frame'].tolist()

# Method 2: Significant Direction Change
hits_direction = df[df['Angle_Change'] > direction_threshold]['Frame'].tolist()

# Method 3: Significant Speed Change
hits_speed = df[df['Speed_Change'] > speed_change_threshold]['Frame'].tolist()

# Method 4: Height Peaks
df_visible = df[df['Visibility'] == 1]
peaks, _ = find_peaks(df_visible['Y'], distance=20, prominence=15)  # Adjusted parameters
hits_height = df_visible.iloc[peaks]['Frame'].tolist()

# Combine hits where multiple criteria are met
combined_hits = set(hits_direction).intersection(hits_speed)
combined_hits = combined_hits.union(set(hits_visibility))
combined_hits = combined_hits.union(set(hits_height))

# Convert to sorted list
combined_hits = sorted(combined_hits)

# Heuristic 1: Apply Minimum Time Gap Between Hits
min_hit_gap = 17.5  # Adjust based on frame rate and game dynamics
final_hits = []
last_hit_frame = -min_hit_gap

for frame in combined_hits:
    if frame - last_hit_frame >= min_hit_gap:
        final_hits.append(frame)
        last_hit_frame = frame

# **Modification Starts Here**

# Remove hits where the previous frame has Visibility == 0
filtered_final_hits = []
for frame in final_hits:
    prev_frame = frame - 1
    # Check if the previous frame exists in the DataFrame
    if prev_frame in df['Frame'].values:
        prev_visibility = df.loc[df['Frame'] == prev_frame, 'Visibility'].values[0]
        if prev_visibility != 0:
            filtered_final_hits.append(frame)
    else:
        # If previous frame is not in DataFrame, include the hit
        filtered_final_hits.append(frame)

# Update final_hits with the filtered hits
final_hits = filtered_final_hits

# **Modification Ends Here**

# Create hits DataFrame
hits_df = pd.DataFrame({'Frame': final_hits, 'Hit': 1})

# Save the hits to a CSV file
hits_df.to_csv('hits.csv', index=False)

print("Detected hit frames have been saved to 'hits.csv'.")