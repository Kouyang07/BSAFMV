import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import os
import csv
import sys

# Court dimensions
court_width, court_length, net_height = 6.1, 13.4, 1.55

def read_court_coordinates(file_path):
    print(f"Reading court coordinates from {file_path}")
    with open(file_path, 'r') as file:
        coordinates = [tuple(map(float, line.strip().split(';'))) for line in file.readlines()[:4]]
    print(f"Court coordinates: {coordinates}")
    return coordinates

def read_shuttle_data(csv_file):
    shuttle_data = {}
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter='\t')
        headers = csv_reader.fieldnames
        for row in csv_reader:
            frame = int(row['Frame'])
            x = float(row['X'])
            y = float(row['Y'])
            visibility = int(row['Visibility']) if 'Visibility' in headers else 1
            shuttle_data[frame] = (x, y, visibility)
    return shuttle_data

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return None
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / det
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / det
    return np.array([px, py])

def calculate_additional_points(corners_2d, corners_3d):
    mid_2d = [(corners_2d[i] + corners_2d[(i+1)%4]) / 2 for i in range(4)]
    mid_3d = [(corners_3d[i] + corners_3d[(i+1)%4]) / 2 for i in range(4)]
    center_2d = line_intersection((*corners_2d[0], *corners_2d[2]), (*corners_2d[1], *corners_2d[3]))
    center_3d = np.mean(corners_3d, axis=0)
    return np.array(mid_2d + [center_2d]), np.array(mid_3d + [center_3d])

def dlt_calibration(points_2d, points_3d):
    A = []
    for (x, y), (X, Y, Z) in zip(points_2d, points_3d):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
    _, _, Vt = np.linalg.svd(np.array(A))
    return Vt[-1].reshape(3, 4)

def dlt_reconstruction(L, point_2d):
    x, y = point_2d
    A = np.array([[L[0,0]-x*L[2,0], L[0,1]-x*L[2,1], L[0,2]-x*L[2,2]],
                  [L[1,0]-y*L[2,0], L[1,1]-y*L[2,1], L[1,2]-y*L[2,2]]])
    b = np.array([x*L[2,3]-L[0,3], y*L[2,3]-L[1,3]])
    return np.linalg.lstsq(A, b, rcond=None)[0]

def calculate_reprojection_error(L, points_2d, points_3d):
    errors = []
    for p2d, p3d in zip(points_2d, points_3d):
        p3d_h = np.append(p3d, 1)
        p2d_proj = L @ p3d_h
        p2d_proj = p2d_proj[:2] / p2d_proj[2]
        errors.append(np.linalg.norm(p2d - p2d_proj))
    return np.mean(errors)

def refine_dlt(L, points_2d, points_3d):
    def objective(params):
        return calculate_reprojection_error(params.reshape(3, 4), points_2d, points_3d)
    result = minimize(objective, L.flatten(), method='Nelder-Mead')
    return result.x.reshape(3, 4)

def process_video(video_path):
    base_name, result_dir = os.path.splitext(os.path.basename(video_path))[0], "result/"
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved in {result_dir}")
    coordinates_file = os.path.join(result_dir, f"{base_name}_court.txt")

    # Read court coordinates
    court_points_2d = np.array(read_court_coordinates(coordinates_file))

    # Define 3D court points
    court_points_3d = np.array([[0, 0, 0], [court_width, 0, 0], [court_width, court_length, 0],
                                [0, court_length, 0], [court_width/2, 0, net_height], [court_width/2, court_length, net_height]])

    # Calculate additional points
    add_points_2d, add_points_3d = calculate_additional_points(court_points_2d, court_points_3d[:4])
    all_points_2d = np.vstack((court_points_2d, add_points_2d))
    all_points_3d = np.vstack((court_points_3d, add_points_3d))

    # Perform DLT calibration
    L = dlt_calibration(all_points_2d, all_points_3d)
    initial_error = calculate_reprojection_error(L, all_points_2d, all_points_3d)
    L_refined = refine_dlt(L, all_points_2d, all_points_3d)
    refined_error = calculate_reprojection_error(L_refined, all_points_2d, all_points_3d)

    print(f"Initial reprojection error: {initial_error}")
    print(f"Refined reprojection error: {refined_error}")

    # Read shuttle data
    shuttle_csv = os.path.join(result_dir, f"{base_name}_shuttle.csv")
    shuttle_data = read_shuttle_data(shuttle_csv)

    # Process each frame
    shuttle_3d_positions = {}
    for frame in range(max(shuttle_data.keys()) + 1):
        if frame in shuttle_data:
            x, y, visibility = shuttle_data[frame]
            shuttle_2d = np.array([x, y])
            shuttle_3d = dlt_reconstruction(L_refined, shuttle_2d)
            shuttle_3d_positions[frame] = (shuttle_3d, visibility)
        else:
            shuttle_3d_positions[frame] = (np.array([np.nan, np.nan, np.nan]), 0)

    # Save 3D coordinates to CSV
    output_csv = os.path.join(result_dir, f"{base_name}_3d.csv")
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'X', 'Y', 'Z', 'Visibility'])
        for frame, (pos, visibility) in shuttle_3d_positions.items():
            writer.writerow([frame, pos[0], pos[1], pos[2], visibility])

    print(f"3D coordinates saved to {output_csv}")

    # Plot the results
    plot_results(all_points_3d, shuttle_3d_positions)

    # Calculate uncertainty
    visible_positions = [pos for pos, vis in shuttle_3d_positions.values() if vis == 1]
    if visible_positions:
        uncertainty = refined_error * np.mean([np.linalg.norm(p) for p in visible_positions]) / np.mean([np.linalg.norm(p) for p in all_points_3d])
        print(f"Estimated uncertainty in shuttle position: ±{uncertainty:.3f} meters")
    else:
        print("No visible shuttle positions to calculate uncertainty")

def plot_results(all_points_3d, shuttle_3d_positions):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([0, court_width, court_width, 0, 0], [0, 0, court_length, court_length, 0], [0, 0, 0, 0, 0], 'b-')
    ax.plot([0, court_width], [court_length/2, court_length/2], [0, 0], 'k-')
    ax.plot([0, court_width], [court_length/2, court_length/2], [net_height, net_height], 'k-')
    ax.plot([0, 0], [court_length/2, court_length/2], [0, net_height], 'k-')
    ax.plot([court_width, court_width], [court_length/2, court_length/2], [0, net_height], 'k-')

    visible_positions = [pos for pos, vis in shuttle_3d_positions.values() if vis == 1]
    if visible_positions:
        shuttle_x, shuttle_y, shuttle_z = zip(*visible_positions)
        ax.scatter(shuttle_x, shuttle_y, shuttle_z, c='r', s=10, label='Shuttle')

    ax.scatter(all_points_3d[:, 0], all_points_3d[:, 1], all_points_3d[:, 2], c='g', s=50, label='Calibration Points')

    ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Z (m)')
    ax.set_title('Badminton Court with Shuttle Trajectory')
    ax.legend()
    max_range = np.array([court_width, court_length, max(net_height, max(shuttle_z) if visible_positions else net_height)]).max()
    ax.set_box_aspect((court_width/max_range, court_length/max_range, 1.0))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 dlt.py <path_to_video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    process_video(video_path)
