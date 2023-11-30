import numpy as np
import cv2
from sklearn.discriminant_analysis import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


import os
import cv2
import numpy as np


open_eyes_path = './archive/train/Open_Eyes'
closed_eyes_path = './archive/train/Closed_Eyes'


X_train = []
y_train = []


label_mapping = {'Closed Eyes': 0, 'Open Eyes': 1}


def load_images_from_folder(folder_path, label):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if img_path.endswith(".png"):
            # Load the image using OpenCV
            img = cv2.imread(img_path)
            # Resize the image to a common size (e.g., 64x64 pixels)
            img = cv2.resize(img, (64, 64))
            # Append the image and label to the lists
            X_train.append(img)
            y_train.append(label)

load_images_from_folder(open_eyes_path, label_mapping['Open Eyes'])
load_images_from_folder(closed_eyes_path, label_mapping['Closed Eyes'])

# Convert the lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train.astype('float32') / 255.0

X_train_shape = X_train.shape
X_train = X_train.reshape(X_train_shape[0], -1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_train = X_train.reshape(X_train_shape)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=42)

dataset_size = len(X_train)


test_size = 0.2

if dataset_size == 0:
    print("Error: The dataset is empty.")
elif dataset_size <= 1 / (1 - test_size):
    print("Warning: The dataset is too small to split with the specified test size.")
    print("Using the entire dataset for training.")
    X_val, y_val = X_train, y_train
else:
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)


img_model = Sequential()
img_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
img_model.add(layers.MaxPooling2D((2, 2)))
img_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
img_model.add(layers.MaxPooling2D((2, 2)))
img_model.add(layers.Flatten())
img_model.add(layers.Dense(64, activation='relu'))
img_model.add(Dropout(0.5))
img_model.add(layers.Dense(32, activation='relu'))
img_model.add(layers.Dense(1, activation='sigmoid'))


img_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = img_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

accuracy = img_model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy[1] * 100:.2f}%")

img_model.save('eyes.keras')

