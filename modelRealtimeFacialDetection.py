import cv2
from keras.preprocessing import image
from keras.utils import img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np

# Load the VGG16 model
model = VGG16(weights='imagenet')

# Define a dictionary for mapping emotion labels to text
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Define a function for predicting the emotion of a frame
def predict_emotion(frame):
    img = cv2.resize(frame, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = model.predict(img)
    emotion_label = np.argmax(preds)
    print("this is it:",emotion_label)

    if emotion_label in emotion_dict:
        emotion_text = emotion_dict[emotion_label]
        return emotion_text  # Add this return statement
    else:
        return "Unknown"  # Add a default return value in case emotion_label is not in emotion_dict

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Define the font for displaying the emotion label
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Predict the emotion of the frame
    emotion_text = predict_emotion(frame)
    
    # Display the emotion label on the frame
    cv2.putText(frame, emotion_text, (50, 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Stop the program if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()






