YOLOv12 Baseball Pitch Tracker

  Utilizing a fine-tuned ultralytics YOLOv12 model, this application acts to extract the frame-by-frame location of a baseball through broadcast footage. The application yields both raw positional data, along with an interpolated overlay tracker.

Dependencies

  The following is a list of requirements for the applications operation:

    - Python 3.0+
    - numpy
    - OpenCV (cv2)
    - PIL
    - Ultralytics

Examples

  Example runs can be found in the examples subfolder.

Training and Results

  A full breakdown of the background, training process, and application results can be found in the Report.pdf. Training scripts can be found in the training subfolder, with results from both training runs being found in the R1 and R2 subfolders respectively.

Usage Instructions

  Video Guidelines: Videos should be trimmed from the ball's release point until the ball reaches the glove ; Videos should be in .mp4 format

  - Ensure python 3.0+ is installed
  - To ensure dependencies, run the command "pip install numpy opencv-python pillow ultralytics"
  - Run command "python Pitch_Tracker.py"
  - Insert path to video which is to be processed (ex. ".../example.mp4")
  - Alternatively, insert "ex1", "ex2", or "ex3" to use example videos
  -  Processing results will be found in the "Outputs" subdirectory, [video title]_detections.mp4
  will have the raw YOLO detections, [video title]_trajectory.mp4 will have the full tracer
  
  
