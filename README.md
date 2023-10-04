# ModelRealTimeFacialDetection
# Emotion Detection Project README

## Introduction

This project implements a real-time emotion detection system using Python. It utilizes the VGG16 deep learning model for image classification to recognize emotions in a live video stream from the default camera.

## Prerequisites

Before running the code, make sure you have the following libraries installed :

- `cv2`: OpenCV for image and video processing.
- `keras`: A high-level neural networks API.
- `numpy`: Fundamental package for numerical computations in Python.

You can install the required packages using the following command:

```bash
pip install opencv-python keras numpy
```

## Code Description

### Libraries Used

- `cv2`: OpenCV is used for capturing video frames, image resizing, and displaying the video stream.
- `keras.preprocessing.image`: This module provides utilities for image preprocessing.
- `keras.utils.img_to_array`: Converts a PIL Image instance to a NumPy array.
- `keras.applications.vgg16.VGG16`: Pre-trained VGG16 model for image classification.
- `numpy`: For numerical operations.

### Emotion Labels

The model predicts emotions from frames. The predicted labels are mapped to text using the `emotion_dict` dictionary.

```python
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
```

### Predicting Emotions

The function `predict_emotion(frame)` takes a frame as input, processes it, and predicts the corresponding emotion. It returns the predicted emotion as text.

```python
def predict_emotion(frame):
    # ... (code for preprocessing and prediction)
```

### Running the Program

Execute the provided script. It captures video from the default camera and displays the emotion label on each frame. Press the 'q' key to exit.

```bash
python emotion_detection.py
```

## Usage and Customization

You can use this code as a starting point for building more complex emotion detection systems. Customize the model, add more labels, or integrate it with other applications.



---

**Author**: [Wala Hassine]

**Contact**: [Walahassine2002@gmail.com]

**Date**: [4/10/2023]
