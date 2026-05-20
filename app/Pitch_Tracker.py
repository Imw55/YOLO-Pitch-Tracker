import math
import os
from pathlib import Path
import numpy as np

import cv2
from PIL import Image, ImageOps

from ultralytics import YOLO

"""
Description: Apply visual and scale transforms to target image in order to replicate training data

frame (numpy array): Image to be processed
return (numpy array): Final processed image
"""
def preprocess(frame):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB format

    img = Image.fromarray(img) # Convert image format

    img = ImageOps.exif_transpose(img)

    img = np.array(img) # Reconvert to np array

    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR) #Strech to 640x640

    # Clahe normalization
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) # Reconvert to BGR

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

"""
Description: Select the single best detection from a frame based on confidence threshold, filtered by a maximum euclidean distance

boxes (ultralytics.yolo.engine.results.Boxes): result.boxes for a single ultralytics YOLO inference
last_x (float): x_value of the previous detection
last_y (float): y_value of the previous detection
max_dist (int): Maximum euclidean distance from last detection for a box to be considered
return (ultralytics.yolo.engine.results.Boxes): Box for best candidate detection
"""
def select_best_box(boxes, last_x, last_y, max_dist=10):

    if boxes is None or len(boxes) == 0: return None # Do nothing if no detections

    sorted_boxes = sorted(boxes, key=lambda b: float(b.conf), reverse=True) # Sort boxes by conf score

    if last_x is None or last_y is None:
        return sorted_boxes[0]

    # Find box with highest confidence score within distance requirement and return
    for box in sorted_boxes:
        cx, cy = box.xywh[0][:2]

        dist = ((cx - last_x)**2 + (cy - last_y)**2) ** 0.5

        if dist <= max_dist:
            return box

    return sorted_boxes[0] # Fallback


"""
Description: Run custom YOLOv12 model over all images of a video

cap (cv2.VideoCapture): OpenCV video capture object for target video
return 1 ([ultralytics.yolo.engine.results.Boxes]): Ordered list of detection boxes by frame
return 2 (numpy array): Ordered array of video frames
return 3 (float): Frames per second of input video
"""
def get_detections(cap):

    # Load model
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "model", "Android_Hernandez.pt")

    model = YOLO(model_path)

    # Get capture data
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 640
    height = 640

    # Create return lists
    det = []
    vid = []
    conf = 0.001 # Conf threshold before first detection (UNUSED)
    first_det = False # First detection flag (UNUSED)

    while True:

        # Get frame, break loop if none remain
        ret, frame = cap.read()
        if not ret: break

        img = preprocess(frame) # Preprocess frame

        # Run inference over frame
        res = model.predict(
            source=img,
            imgsz=640,
            conf=conf,
            verbose=False
        )

        # Check for first detection, lower confidence threshold (UNUSED
        if res is not None and not first_det:
            conf = 0.001
            first_det = True

        # Append to return lists
        det.append(res)
        vid.append(img)

    return det,np.array(vid),fps


"""
Description: Interpolates a trajectory of an object based on a list of filtered detections

det ([ultralytics.yolo.engine.results.Boxes]): Ordered list of best detections by frame
vid ([numpy array]): Ordered list of frames represented as numpy arrays
fps (float): Frames per second of video
return 1 (float): Ordered list of x coordinates for the centres of best detections
return 2 (float): Ordered list of y coordinates for the centres of best detections
"""
def get_trajectory(det, vid, fps):

    # Create return lists and last detection variables
    x = []
    y = []
    last_x, last_y = None, None

    # For every frames detections, get the best one
    for i, frame in enumerate(det):

        # Fallback for no detection
        if frame is None or len(frame) == 0:
            x.append(None)
            y.append(None)
            continue

        result = frame[0]

        # If not detections, add empty entry
        if result.boxes is None or len(result.boxes) == 0:
            x.append(None)
            y.append(None)
            continue
        
        best = select_best_box(result.boxes, last_x, last_y) # Get best detection

        # Fallback for no good detetions
        if best is None:
            x.append(None)
            y.append(None)
            continue
        
        # Add to return list
        coords = best.xywh[0]

        x_center = float(coords[0])
        y_center = float(coords[1])

        x.append(x_center)
        y.append(y_center)

    last_x, last_y = x_center, y_center

    # Loop variables
    first_found = False
    gaps = []
    i = 0

    # Find continous sections of empty entries
    while i < len(x):

        # Do not log initial detection gap
        if x[i] is None and first_found:

            start = i
            end = i

            # Get first and last index of gap, append it to list as a tuple
            while end < len(x) and x[end] is None: end += 1
            if start > 0 and end < len(x): gaps.append((start, end - 1))

            i = end
        
        # First detection check
        else:
            if x[i] is not None:
                first_found = True
            i += 1

    # For every gap, interpolate
    for pair in gaps:

        start, end = pair # Get indexes

        num_frames = end - start + 1 # Get total number of frames

        x0 = x[start - 1]
        x1 = x[end + 1]
        y0 = y[start - 1]
        y1 = y[end + 1]

        # Divide xy gap evenly among all mssing frames
        step_x = (x1 - x0) / (num_frames + 1) 
        step_y = (y1 - y0) / (num_frames + 1)

        # Add linearly interpolated points
        for k in range(1, num_frames + 1):
            x[start + k - 1] = x0 + step_x * k
            y[start + k - 1] = y0 + step_y * k

    return x,y


"""
Description: Get pitch metrics based on an interpolated trajectory

x ([float]): x coordinate of ball at given frames
y ([float]): y coordinate of ball at given frames
fps (float): FPS of analyzed video
dist (float): Presumed pitch travel distance (in ft)
"""
def get_metrics(x, y, fps, dist=55.5):

    num_frames = sum(1 for item in x if item is not None) # Get duration of pitch in frames
    dt = (num_frames+3)/fps # Calculate duration of pitch in seconds
    feet_ps = dist/dt # Calculate speed in ft/s
    mph = feet_ps/1.47 # Convert to mph
    print("Velocity: ",round(mph,2),"mph")


"""
Description: Add overlay to original video and save it to desired path

vid ([numpy array]): Numpy array representing frames of the video
x ([float]): x coordinate of ball at given frames
y ([float]): y coordinate of ball at given frames
out_path (str): Desired system path for final video
"""
def draw_and_save_trajectory(vid, x, y, fps, out_path):

    # Check trajectory exists
    if x is None:
        print("No trajectory")
        return

    # Get video writer
    h, w, _ = vid[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Draw trajectory on frame
    for i, frame in enumerate(vid):

        img = frame.copy()

        for k in range(min(i+1, len(x))):

            if x[k] is None: continue
            if 0 <= x[k] < w and 0 <= y[k] < h: cv2.circle(img, (int(x[k]), int(y[k])), 3, (0, 255, 0), -1)

        writer.write(img)

    writer.release()

"""
Description: Overlay all detections on original video and save to desired path

vid ([numpy array]): Numpy array representing frames of the video
det ([ultralytics.yolo.engine.results.Boxes]): Ordered list of best detections by frame
out_path (str): Desired system path for final video
"""
def draw_and_save_detections(vid, det, fps, out_path):

    # Check for empty video
    if vid is None or len(vid) == 0:
        print("No video frames")
        return

    # Get Writer
    h, w, _ = vid[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Draw boxes on each frame of video
    for i in range(len(vid)):

        if det[i] is None or len(det[i]) == 0:
            writer.write(vid[i])
            continue

        result = det[i][0]

        annotated = result.plot()

        writer.write(annotated)

    writer.release()
"""
Description: Get user inputed video or use example to draw a pitch overlay, show all model detections, and show metrics for a given pitch
"""
def main():

    in_path = input("Please enter video path (input ex1, ex2, or ex3 for examples): ") # Get user video path

    base_dir = os.path.dirname(__file__)

    # Check if example was selected, replace path with appropriate path
    if in_path == "ex1": in_path = os.path.join(base_dir, "Example_Inputs", "ex1.mp4")
    elif in_path == "ex2": in_path = os.path.join(base_dir, "Example_Inputs", "ex2.mp4")
    elif in_path == "ex3": in_path = os.path.join(base_dir, "Example_Inputs", "ex3.mp4")
    elif in_path == "ex4": in_path = os.path.join(base_dir, "Example_Inputs", "ex4.mp4")

    file_name = Path(in_path).stem

    output_dir = os.path.join(base_dir, "Outputs")

    # Get appropriate output paths 
    out_path_detections = os.path.join(output_dir, file_name + "_detections.mp4")
    out_path_trajectory = os.path.join(output_dir, file_name + "_trajectory.mp4")
 
    # Run functions
    cap = cv2.VideoCapture(in_path)
    print("Generating Detections...")
    det, vid, fps = get_detections(cap)
    print("Saving Detections...")
    draw_and_save_detections(vid,det,fps,out_path_detections)
    print("Generating Trajectory...")
    x,y = get_trajectory(det, vid, fps)
    print("Saving Trajectory...")
    draw_and_save_trajectory(vid,x,y,fps,out_path_trajectory)
    get_metrics(x,y,fps)


main()
