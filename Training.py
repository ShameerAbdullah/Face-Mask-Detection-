import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset directory
dataset_dir = 'C:/Users/Shameer/Downloads/Mask Detection'

# Define image dimensions
img_width, img_height = 100, 100

# Prepare dataset for training
classes = ['mask', 'no_mask']
num_classes = len(classes)

data = []
labels = []

# Read images and labels from dataset directory
for label, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        
        # Load and resize the image
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (img_width, img_height))
            data.append(image)
            labels.append(label)
        else:
            print(f"Failed to load image: {image_path}")

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize pixel values to the range of 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoded vectors
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training set
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

# Train the model
batch_size = 32
epochs = 10

model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train) // batch_size,
          epochs=epochs, validation_data=(X_test, y_test))

# Save the trained model
model.save('mask_detection_model.h5')
