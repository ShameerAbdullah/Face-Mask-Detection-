import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = tf_models.load_model('C:/Users/Shameer/Downloads/Mask Detection/mask_detector.model')

# Define the function to detect masks
def detect_mask(frame):
    # Preprocess the frame
    resized = cv2.resize(frame, (224, 224))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 224, 224, 3))

    # Perform mask detection
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]

    # Add bounding box and label to the frame
    if label == 0:
        cv2.putText(frame, 'Mask', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No Mask', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Open the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        break

    # Perform mask detection on the frame
    frame = detect_mask(frame)

    # Display the resulting frame
    cv2.imshow('Mask Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video_capture.release()
cv2.destroyAllWindows()

# Load the pre-trained model
#model = tf.keras.models.load_model('C:/Users/Shameer/Downloads/Mask Detection/mask_detection_model.h5') #tf.keras.models.load_model('path_to_your_model')  # Replace with the actual file path

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the labels for mask and no mask
labels = {0: 'Mask', 1: 'No Mask'}

# Function to detect face and classify mask
def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized = cv2.resize(face_roi, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        colored = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)  # Convert grayscale to color
        reshaped_colored = np.reshape(colored, (1, 100, 100, 3))  # Reshape to match model's input shape
        result = model.predict(reshaped_colored)

        label = np.argmax(result, axis=1)[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = detect_mask(frame)

    cv2.imshow('Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
