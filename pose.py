import sys
import csv
import os
from ultralytics import YOLO
import cv2
import numpy as np

def load_processed_frames(processed_csv_file):
    with open(processed_csv_file, 'r') as csvfile:
        return {int(row['frame']): int(row['is_rally_scene'] == 'True') for row in csv.DictReader(csvfile)}

def read_court_coordinates(file_path):
    print(f"Reading court coordinates from {file_path}")
    with open(file_path, 'r') as file:
        coordinates = [tuple(map(float, line.strip().split(';'))) for line in file.readlines()[:4]]
    print(f"Court coordinates: {coordinates}")
    return coordinates

def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def box_intersects_court(box, court_coords):
    x1, y1, x2, y2 = box
    return any(point_inside_polygon(x, y, court_coords) for x, y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)])


def process_video(video_path):
    # Load the YOLOv10 model
    model = YOLO("yolov10x.pt")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the base name of the video file
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create the result directory if it doesn't exist
    os.makedirs("result", exist_ok=True)

    # Prepare the CSV file
    csv_path = f"result/{base_name}_pose.csv"

    # Prepare the output video
    out_path = f"result/{base_name}_pose.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Load processed frames
    preprocessed_path = f"result/{base_name}_preprocessed.csv"
    rally_frames = load_processed_frames(preprocessed_path)

    # Load court coordinates
    court_path = f"result/{base_name}_court.txt"
    court_coords = read_court_coordinates(court_path)

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'Class', 'Confidence', 'X', 'Y', 'Width', 'Height'])

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Check if it's a rally frame
            if rally_frames.get(frame_number, 0) == 1:
                # Perform detection
                results = model(frame)

                # Process and write results to CSV
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Check if the detected object is a person
                        if box.cls == 0:  # Assuming class 0 is 'person'
                            x1, y1, x2, y2 = box.xyxy[0]

                            # Check if the bounding box intersects with or is inside the court
                            if box_intersects_court((x1, y1, x2, y2), court_coords):
                                x, y, w, h = box.xywh[0]
                                conf = box.conf[0]
                                csvwriter.writerow([frame_number, 'person', conf.item(), x.item(), y.item(), w.item(), h.item()])

                                # Draw bounding box on the frame
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(frame, f'Person: {conf:.2f}', (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Draw court lines
                court_poly = np.array(court_coords, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [court_poly], True, (255, 0, 0), 2)

            # Write the frame to the output video
            out.write(frame)

            frame_number += 1

    cap.release()
    out.release()
    print(f"Detection results saved to {csv_path}")
    print(f"Annotated video saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 pose.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    process_video(video_path)
