import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the 3D world points of the badminton court
def define_world_points():
    world_points = np.array([
        [0, 0, 0], [0, 13.4, 0], [6.1, 13.4, 0], [6.1, 0, 0],  # P1, P2, P3, P4
        [0.46, 0, 0], [0.46, 13.4, 0], [5.64, 13.4, 0], [5.64, 0, 0],  # P5, P6, P7, P8
        [0, 4.715, 0], [6.1, 4.715, 0], [0, 8.68, 0], [6.1, 8.68, 0],  # P9, P10, P11, P12
        [3.05, 4.715, 0], [3.05, 8.68, 0], [0, 6.695, 0], [6.1, 6.695, 0],  # P13, P14, P15, P16
        [0, 0.76, 0], [6.1, 0.76, 0], [0, 12.64, 0], [6.1, 12.64, 0],  # P17, P18, P19, P20
        [3.05, 0, 0], [3.05, 13.4, 0]])  # P21, P22
    return world_points

# Read the 2D court coordinates from the CSV file
def read_court_points(court_csv):
    try:
        df = pd.read_csv(court_csv)
        df = df[~df['Point'].str.contains('NetPole')]
        return df[['X', 'Y']].values
    except FileNotFoundError:
        print(f"Error: File {court_csv} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File {court_csv} is empty.")
        sys.exit(1)

# Perform intrinsic calibration
def intrinsic_calibration(world_points, image_points):
    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros(5)

    world_points = np.array([world_points.astype(np.float32)]).reshape((22, 1, 3))
    image_points = np.array([image_points.astype(np.float32)]).reshape((22, 1, 2))

    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera([world_points], [image_points], (1920, 1080), None, None, flags=flags)

    print("Intrinsic Parameters:")
    print("Camera Matrix:")
    print(camera_matrix)
    print("Distortion Coefficients:")
    print(dist_coeffs)

    return camera_matrix, dist_coeffs

# Perform extrinsic calibration
def extrinsic_calibration(camera_matrix, dist_coeffs, world_points, image_points):
    ret, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)

    print("Extrinsic Parameters:")
    print("Rotation Vector:")
    print(rvec)
    print("Translation Vector:")
    print(tvec)

    return rvec, tvec

# Plot the camera's position and the court
def plot_camera_and_court(world_points, rvec, tvec):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the court
    # Plot the court
    ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], c='b')

    # Plot the camera's position
    camera_position = np.array([tvec[0], tvec[1], tvec[2]])
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', marker='x')

    # Plot the camera's orientation
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_orientation = np.dot(rotation_matrix, np.array([0, 0, 1]))
    ax.quiver(camera_position[0], camera_position[1], camera_position[2], camera_orientation[0], camera_orientation[1], camera_orientation[2], color='r')

    # Set plot limits and labels
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 20)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    plt.show()

# Main function
def main():
    world_points = define_world_points()

    court_csv = 'result/test_court.csv'
    image_points = read_court_points(court_csv)

    camera_matrix, dist_coeffs = intrinsic_calibration(world_points, image_points)

    rvec, tvec = extrinsic_calibration(camera_matrix, dist_coeffs, world_points, image_points)

    plot_camera_and_court(world_points, rvec, tvec)

if __name__ == "__main__":
    main()