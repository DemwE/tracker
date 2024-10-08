import cv2
import torch
from ultralytics import YOLO
import os
import sys
import argparse
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parsing
parser = argparse.ArgumentParser(description='YOLOv8 Object Detection with ROI')
parser.add_argument('--model', type=str, default='yolov8m.pt', help='Path to YOLOv8 model file')
parser.add_argument('--camera', type=int, default=1, help='Camera index for video capture')
parser.add_argument('--width', type=int, default=1280, help='Frame width')
parser.add_argument('--height', type=int, default=720, help='Frame height')
parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detections')
args = parser.parse_args()

# Check if model file exists
if not os.path.exists(args.model):
    logging.error(f"Model file '{args.model}' not found.")
    sys.exit(1)

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Load and move model to device
model = YOLO(args.model)
model.to(device)
logging.info("Model loaded successfully.")

# Initialize video capture
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    logging.error("Error: Could not open camera.")
    sys.exit(1)
logging.info("Video capture started.")

# Set video resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
if actual_width != args.width or actual_height != args.height:
    logging.warning(f"Requested resolution {args.width}x{args.height} not supported. Using {actual_width}x{actual_height}.")

# Define the initial region of interest (ROI) coordinates
roi_x1, roi_y1, roi_x2, roi_y2 = 100, 100, 600, 400  # Example coordinates
drawing = False  # True if mouse is pressed
resizing = False  # True if resizing
moving = False  # True if moving
ix, iy = -1, -1

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, roi_x1, roi_y1, roi_x2, roi_y2, drawing, resizing, moving

    if event == cv2.EVENT_LBUTTONDOWN:
        if roi_x1 - 10 < x < roi_x1 + 10 and roi_y1 < y < roi_y2:
            resizing = 'left'
        elif roi_x2 - 10 < x < roi_x2 + 10 and roi_y1 < y < roi_y2:
            resizing = 'right'
        elif roi_y1 - 10 < y < roi_y1 + 10 and roi_x1 < x < roi_x2:
            resizing = 'top'
        elif roi_y2 - 10 < y < roi_y2 + 10 and roi_x1 < x < roi_x2:
            resizing = 'bottom'
        elif roi_x1 < x < roi_x2 and roi_y1 < y < roi_y2:
            moving = True
            ix, iy = x, y
        else:
            drawing = True
            ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_x1, roi_y1, roi_x2, roi_y2 = ix, iy, x, y
        elif resizing:
            if resizing == 'left':
                roi_x1 = max(0, x)
            elif resizing == 'right':
                roi_x2 = min(actual_width, x)
            elif resizing == 'top':
                roi_y1 = max(0, y)
            elif resizing == 'bottom':
                roi_y2 = min(actual_height, y)
        elif moving:
            dx, dy = x - ix, y - iy
            new_x1 = roi_x1 + dx
            new_x2 = roi_x2 + dx
            new_y1 = roi_y1 + dy
            new_y2 = roi_y2 + dy
            if new_x1 < 0:
                new_x1 = 0
                new_x2 = roi_x2 - roi_x1
            if new_x2 > actual_width:
                new_x2 = actual_width
                new_x1 = actual_width - (roi_x2 - roi_x1)
            if new_y1 < 0:
                new_y1 = 0
                new_y2 = roi_y2 - roi_y1
            if new_y2 > actual_height:
                new_y2 = actual_height
                new_y1 = actual_height - (roi_y2 - roi_y1)
            roi_x1, roi_x2 = new_x1, new_x2
            roi_y1, roi_y2 = new_y1, new_y2
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        resizing = False
        moving = False

cv2.namedWindow('Full Camera View')
cv2.setMouseCallback('Full Camera View', draw_rectangle)

# Initialize FPS calculation
start_time = time.time()
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        logging.error("Error: Could not read frame.")
        break

    # Retrieve the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]

    # Ensure that the indices are within the valid range and cast to integers
    roi_y1 = int(max(0, min(roi_y1, frame_height)))
    roi_y2 = int(max(0, min(roi_y2, frame_height)))
    roi_x1 = int(max(0, min(roi_x1, frame_width)))
    roi_x2 = int(max(0, min(roi_x2, frame_width)))

    # Ensure ROI is valid
    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
        # Crop the frame to the ROI
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Perform object detection
        results = model(roi_frame)

        # Draw bounding boxes and labels on the cropped frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls)]
                score = float(box.conf)  # Convert tensor to float
                if score > args.threshold:
                    cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(roi_frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the cropped frame with detected objects
        cv2.imshow('Detection Square', roi_frame)

    # Draw the ROI rectangle on the full frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    else:
        fps = frame_count / elapsed_time

    # Calculate position for the FPS counter in the bottom-right corner
    text = f'FPS: {fps:.2f}'
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    x = frame.shape[1] - text_width - 10
    y = frame.shape[0] - baseline - 10

    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the full camera view
    cv2.imshow('Full Camera View', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close the windows
cap.release()
cv2.destroyAllWindows()