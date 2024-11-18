# Lidar Camera Fusion for Depth Estimation and Object Detection

This project implements a multi-modal object detection pipeline using LIDAR and Camera data from the KITTI dataset. It combines object detection and tracking to achieve accurate 3D localization and object tracking in real-time, which is useful for autonomous vehicle systems.

## Overview

The goal of this project is to demonstrate a multi-sensor fusion approach where LIDAR (Light Detection and Ranging) data is integrated with images from a camera to enhance the precision and robustness of object detection in autonomous systems. The project uses state-of-the-art deep learning models for 2D object detection (e.g., YOLO, Faster R-CNN) and combines this with LIDAR data to estimate the 3D locations of detected objects.
Key Features:
- Object Detection on KITTI Dataset: Uses pretrained models to detect objects (e.g., cars, pedestrians, cyclists) in 2D images.
- LIDAR Point Cloud Processing: Extracts depth information from LIDAR point clouds and fuses it with camera data to obtain 3D object locations.
- Real-time Tracking: Implements tracking algorithms to keep track of detected objects across frames.
- Visualization: Visualizes 2D bounding boxes, 3D object locations, and LIDAR point clouds in real-time.

## Setup & Installation
Prerequisites

To run this notebook and its associated code, you will need the following dependencies installed:
- Python 3.x
- TensorFlow or PyTorch (depending on the model you are using)
- OpenCV for image and video processing
- ROS (Robot Operating System) for sensor fusion and real-time communication (optional)
- NumPy, Matplotlib, and other common scientific computing libraries
- KITTI dataset (available here)

## This notebook contains the following steps:
- Data Preprocessing: Loads and preprocesses KITTI images and LIDAR point clouds.
- Object Detection: Uses a pre-trained detection model (e.g., YOLO) to detect objects in 2D images.
- LIDAR Fusion: Fuses 2D detections with 3D LIDAR point clouds to estimate object locations in 3D space.
- Tracking: Implements a tracking algorithm to maintain object IDs across frames.
- Visualization: Visualizes both 2D detections and 3D tracking in the environment.
