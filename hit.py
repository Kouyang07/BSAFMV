import sys
import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

def load_court_coordinates(base_name):
    court_file = f"result/{base_name}_court.txt"
    with open(court_file, 'r') as f:
        court_coords = [list(map(float, line.strip().split(';'))) for line in f]
    return court_coords

def load_shuttle_data(base_name):
    shuttle_file = f"result/{base_name}_shuttle.csv"
    df = pd.read_csv(shuttle_file)
    return df[df['Visibility'] != 0].reset_index(drop=True)

def smooth_trajectory(df, window_length=5, polyorder=2):
    df['X_smooth'] = savgol_filter(df['X'], window_length, polyorder)
    df['Y_smooth'] = savgol_filter(df['Y'], window_length, polyorder)
    return df

def calculate_velocity_acceleration(df):
    df['vx'] = df['X_smooth'].diff() / df['Frame'].diff()
    df['vy'] = df['Y_smooth'].diff() / df['Frame'].diff()
    df['v'] = np.sqrt(df['vx']**2 + df['vy']**2)

    df['ax'] = df['vx'].diff() / df['Frame'].diff()
    df['ay'] = df['vy'].diff() / df['Frame'].diff()
    df['a'] = np.sqrt(df['ax']**2 + df['ay']**2)

    return df

def estimate_trajectory(df, window_size=30):
    df['estimated_y'] = np.nan
    df['trajectory_deviation'] = np.nan

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    ransac = RANSACRegressor(random_state=42)

    for i in range(len(df)):
        start = max(0, i - window_size)
        end = min(len(df), i + 1)
        window = df.iloc[start:end]

        try:
            X = window['X_smooth'].values.reshape(-1, 1)
            y = window['Y_smooth'].values

            X_poly = poly_features.fit_transform(X)
            ransac.fit(X_poly, y)

            X_predict = poly_features.transform([[df.loc[i, 'X_smooth']]])
            df.loc[i, 'estimated_y'] = ransac.predict(X_predict)[0]
            df.loc[i, 'trajectory_deviation'] = abs(df.loc[i, 'Y_smooth'] - df.loc[i, 'estimated_y'])
        except:
            pass

    return df

def detect_hits(df, court_coords, velocity_threshold=50, acceleration_threshold=200, min_frames_between_hits=10, window_size=5):
    net_y = (court_coords[4][1] + court_coords[5][1]) / 2  # Average y-coordinate of net top

    hits = []
    last_hit_frame = -np.inf

    print("\nDetailed hit detection process:")
    print("=" * 50)

    for i in range(window_size, len(df) - window_size):
        curr_row = df.iloc[i]
        window = df.iloc[i-window_size:i+window_size+1]

        print(f"\nAnalyzing frame {curr_row['Frame']}:")
        print(f"  Position: ({curr_row['X_smooth']:.2f}, {curr_row['Y_smooth']:.2f})")
        print(f"  Velocity: {curr_row['v']:.2f}")
        print(f"  Acceleration: {curr_row['a']:.2f}")

        # Check if the shuttle crosses the net
        net_crossed = (df.iloc[i-1]['Y_smooth'] - net_y) * (curr_row['Y_smooth'] - net_y) <= 0
        print(f"  Net crossed: {net_crossed}")

        # Check time since last hit
        frames_since_last_hit = curr_row['Frame'] - last_hit_frame
        print(f"  Frames since last hit: {frames_since_last_hit}")

        hit_detected = False
        if frames_since_last_hit > min_frames_between_hits:
            # Detect sudden changes in velocity and direction
            velocity_change = max(window['v']) - min(window['v'])
            direction_change = np.abs(np.diff(np.arctan2(np.diff(window['Y_smooth']), np.diff(window['X_smooth'])))).max()

            if velocity_change > velocity_threshold and curr_row['a'] > acceleration_threshold and direction_change > 0.5:
                hit_detected = True
                print("  HIT DETECTED")
            else:
                print("  No hit detected")

        if hit_detected:
            hits.append({
                'frame': curr_row['Frame'],
                'x': curr_row['X_smooth'],
                'y': curr_row['Y_smooth'],
                'velocity': curr_row['v'],
                'acceleration': curr_row['a'],
                'net_crossed': net_crossed
            })
            last_hit_frame = curr_row['Frame']

    return hits

def main(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    court_coords = load_court_coordinates(base_name)
    df = load_shuttle_data(base_name)

    print(f"Loaded {len(df)} visible shuttle positions")

    df = smooth_trajectory(df)
    df = calculate_velocity_acceleration(df)

    hits = detect_hits(df, court_coords, velocity_threshold=50, acceleration_threshold=200, min_frames_between_hits=10, window_size=5)

    print("\nSummary:")
    print(f"Detected {len(hits)} hits:")
    for hit in hits:
        print(f"Frame: {hit['frame']}, X: {hit['x']:.2f}, Y: {hit['y']:.2f}, Velocity: {hit['velocity']:.2f}, "
              f"Acceleration: {hit['acceleration']:.2f}, Net Crossed: {hit['net_crossed']}")

    # Save hit frames to file
    hit_frames = [str(int(hit['frame'])) for hit in hits]
    hit_file_path = f"result/{base_name}_hit.txt"
    with open(hit_file_path, 'w') as f:
        f.write(','.join(hit_frames))

    print(f"\nHit frames saved to {hit_file_path}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 hit.py samples/test.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    main(video_path)