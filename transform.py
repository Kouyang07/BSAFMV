import cv2
import numpy as np
pts_src = np.array([[273.146, 386.079], [135.366, 703.384], [966.518, 701.338], [799.461, 387.37]])
pts_dst = np.array([[0, 0], [0, 13.4], [6.1, 13.4], [6.1, 0]])
h, _ = cv2.findHomography(pts_src, pts_dst)
point_src = np.array([[745, 342]], dtype=np.float32)
point_src = point_src.reshape(-1, 1, 2)

point_dst = cv2.perspectiveTransform(point_src, h)

print(point_dst)