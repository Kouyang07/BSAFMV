import cv2
import numpy as np
import pandas as pd

# Define the source and destination points
pts_src = np.array([[273.146, 386.079], [135.366, 703.384], [966.518, 701.338], [799.461, 387.37]])
pts_dst = np.array([[0, 0], [0, 13.4], [6.1, 13.4], [6.1, 0]])

# Find the homography matrix
h, _ = cv2.findHomography(pts_src, pts_dst)

# Read the CSV file
df = pd.read_csv('result/test3_shuttle.csv')

# Apply the transformation to each frame
for index, row in df.iterrows():
    point_src = np.array([[row['X'], row['Y']]], dtype=np.float32)
    point_src = point_src.reshape(-1, 1, 2)
    point_dst = cv2.perspectiveTransform(point_src, h)
    x = float(point_dst[0][0][0])  # Explicitly cast to float
    y = float(point_dst[0][0][1] + 13.4)  # Explicitly cast to float
    df.at[index, 'X'] = x
    df.at[index, 'Y'] = y

df.to_csv('result/test3_shuttle_transformed.csv', index=False)

# Create a black background image
court = np.zeros((1340+200, 610+200, 3), np.uint8)  # Added 3 channels for color

# Draw the badminton court
cv2.line(court, (100, 100), (100, 1440), (255, 255, 255), 2)  # Left line
cv2.line(court, (710, 100), (710, 1440), (255, 255, 255), 2)  # Right line
cv2.line(court, (100, 100), (710, 100), (255, 255, 255), 2)  # Bottom line
cv2.line(court, (100, 1440), (710, 1440), (255, 255, 255), 2)  # Top line
cv2.line(court, (100, 770), (710, 770), (255, 255, 255), 1)  # Net

# Create a video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
out = cv2.VideoWriter('badminton_video.mp4', fourcc, fps, (610+200, 1340+200))  # Swapped width and height

# Draw the shuttle positions on the court
for index, row in df.iterrows():
    court_copy = court.copy()
    cv2.putText(court_copy, f"Frame: {int(row['Frame'])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    print(row['X']*100+100, row['Y']*100+100)
    cv2.circle(court_copy, (int(row['X']*100+100), int(row['Y']*100+100)), 5, (0, 255, 0), -1)
    out.write(court_copy)

# Release the video writer
out.release()