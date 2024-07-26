from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x-pose.pt")  # load an official model

# Predict with the model
results = model("samples/test.mp4", show=True, conf=0.3, save=False)# predict on an image