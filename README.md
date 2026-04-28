# Real-Time Pothole Detection

A real-time computer vision application that detects road potholes from uploaded or live road videos using **YOLOv8, OpenCV, Flask, CUDA, and NVIDIA TensorRT**.

This project demonstrates how deep learning-based object detection can be used for road safety, infrastructure monitoring, and automated defect detection in transportation systems.

---

## Overview

Road potholes are a major safety and maintenance issue. Manual road inspection can be slow, expensive, and difficult to scale across large cities or highways.

This project builds a real-time pothole detection dashboard that allows users to upload road videos, run object detection on each frame, and visualize detected potholes with bounding boxes and confidence scores.

The system uses a YOLOv8 object detection model and is designed for GPU-accelerated inference using CUDA and TensorRT optimization.

---

## Problem Statement

City infrastructure teams and road maintenance departments need faster ways to identify potholes and prioritize repairs.

Traditional inspection methods often require manual surveys, delayed reporting, and high operational effort.

This project answers questions such as:

- Can potholes be detected automatically from road video footage?
- Can object detection models identify potholes in real time?
- How can detection confidence and FPS be monitored during inference?
- How can a lightweight web dashboard make the model easier to use?
- How can GPU optimization improve detection performance?

---

## Key Features

- Real-time pothole detection from road videos
- YOLOv8-based object detection
- Flask web dashboard for video upload and visualization
- OpenCV-based frame processing
- Bounding box visualization for detected potholes
- Confidence score display
- FPS monitoring during inference
- CUDA-enabled GPU inference support
- TensorRT FP16 optimization support
- Utility script for converting segmentation masks to YOLO bounding box format
- Simple HTML template-based dashboard

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Web Framework | Flask |
| Computer Vision | OpenCV |
| Object Detection | YOLOv8 |
| Deep Learning | PyTorch |
| Model Optimization | NVIDIA TensorRT |
| GPU Acceleration | CUDA |
| Frontend | HTML |
| Model Library | Ultralytics |

---

## Repository Structure

real-time-pothole-detection/
│
├── templates/
│   └── HTML templates for the Flask dashboard
│
├── app.py
│   └── Main Flask application for video upload and pothole detection
│
├── convert_masks_to_yolo_boxes.py
│   └── Utility script to convert mask annotations into YOLO bounding box format
│
└── README.md

## System Workflow

    Road Video Input
            ↓
    Upload Through Flask Dashboard
            ↓
    Frame Extraction Using OpenCV
            ↓
    YOLOv8 Pothole Detection
            ↓
    Bounding Box + Confidence Score Generation
            ↓
    FPS Monitoring
            ↓
    Annotated Video Output / Live Visualization
    
## How It Works
**1. Video Input**

The user uploads a road video through the Flask dashboard.

The video may contain road surfaces captured from a vehicle, mobile camera, or dashcam-like perspective.

**2. Frame Processing**

OpenCV reads the video frame by frame.

Each frame is passed into the YOLOv8 model for pothole detection.

**3. Object Detection**

The YOLOv8 model identifies potholes and returns:

Bounding box coordinates
Class label
Detection confidence score

**4. Visualization**

Detected potholes are highlighted on the video frame using bounding boxes.

The dashboard displays the processed output so users can visually inspect detections.

**5. Performance Monitoring**

The application can monitor inference speed using FPS.

This helps evaluate whether the model is suitable for near real-time deployment.

## Model Optimization

This project is designed with GPU acceleration in mind.

Supported optimization concepts include:

CUDA-based inference
TensorRT model conversion
FP16 precision optimization
Faster frame-level inference
Reduced latency for real-time detection

TensorRT optimization is especially useful when deploying object detection models on NVIDIA GPUs.

## Annotation Utility

The repository includes:

convert_masks_to_yolo_boxes.py

This script can be used to convert segmentation mask-style annotations into YOLO bounding box format.

YOLO annotation format usually follows:

class_id x_center y_center width height

where coordinates are normalized between 0 and 1.

This is useful when preparing custom pothole datasets for YOLO training.

## Example Use Case

A city maintenance team could mount a camera on a vehicle and record road footage.

The system can process the video and detect potholes automatically, helping teams:

Identify damaged roads faster
Reduce manual inspection effort
Prioritize high-risk areas
Build datasets for road maintenance planning
Support smart city infrastructure monitoring

## Getting Started

**1. Clone the Repository**
git clone https://github.com/Shreevikas-BJ/real-time-pothole-detection.git
cd real-time-pothole-detection
**2. Create a Virtual Environment**
python -m venv venv

Activate the environment:

Windows

venv\Scripts\activate

macOS / Linux

source venv/bin/activate

**3. Install Dependencies**

If a requirements.txt file is available:

pip install -r requirements.txt

If not, install the main dependencies manually:

pip install flask opencv-python ultralytics torch torchvision numpy

For GPU acceleration, make sure your PyTorch installation matches your CUDA version.

## Run the Application

python app.py

After running the command, open the local Flask URL in your browser.

Usually:

http://127.0.0.1:5000/

Upload a road video and run pothole detection through the dashboard.

## YOLOv8 Model File

If the model file is not included in the repository, place your trained YOLOv8 pothole detection model in the project folder.

Example model file names:

    best.pt
    yolov8_pothole.pt
    pothole_detector.pt

Then update the model path inside app.py if required.

Example:

    model = YOLO("best.pt")

## Dataset Preparation

If you are preparing a custom pothole dataset, make sure the dataset follows YOLO format:

    dataset/
    │
    ├── images/
    │   ├── train/
    │   └── val/
    │
    ├── labels/
    │   ├── train/
    │   └── val/
    │
    └── data.yaml

Each label file should contain bounding box annotations in this format:

class_id x_center y_center width height

## Recommended Repository Description

Real-time computer vision app for detecting road potholes from video streams using YOLOv8, TensorRT, CUDA, OpenCV, and Flask.

## Business and Social Value

This project demonstrates how computer vision can be applied to real-world infrastructure problems.

Potential impact areas include:

Road safety improvement
Smart city monitoring
Automated infrastructure inspection
Faster road maintenance planning
Reduced manual survey cost
Computer vision-based public works analytics

## Key Learnings

This project helped strengthen:

Object detection using YOLOv8
Real-time video processing with OpenCV
Flask-based ML application development
Bounding box visualization
Computer vision inference workflow
CUDA and TensorRT optimization concepts
Dataset annotation conversion
Deployment-style thinking for vision models

## Future Improvements

Add a requirements.txt file for easier setup
Add sample test video
Add trained YOLOv8 model download instructions
Add dashboard screenshots or demo GIF
Add GPS metadata support for pothole location tracking
Add severity classification for potholes
Add database storage for detected pothole events
Add map visualization for detected pothole locations
Add Docker support
Add cloud deployment instructions
Add batch processing for multiple videos
Add model performance metrics such as mAP, precision, recall, and FPS

## Disclaimer

This project is built for educational and portfolio purposes. Real-world road inspection systems should be validated with diverse road conditions, lighting environments, camera angles, and safety requirements before production deployment.

## Author

Shreevikas Bangalore Jagadish
Graduate Student, Information Technology and Management
Illinois Institute of Technology

GitHub: Shreevikas-BJ
LinkedIn: shreevikasbj
Portfolio: datascienceportfol.io/shreevikasbj

## Repository

    https://github.com/Shreevikas-BJ/real-time-pothole-detection
