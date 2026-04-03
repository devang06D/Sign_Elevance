import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

BASE_PATH = r"D:\projects Elevance\sign\isl_dataset"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR  = os.path.join(BASE_PATH, "test")

class_folders = sorted([f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))])
print("Detected 36 classes:", class_folders)
print("Total classes:", len(class_folders))

def load_images(folder_path, img_size=(100, 100)):
    images = []
    labels = []
    for label, class_name in enumerate(class_folders):
        class_path = os.path.join(folder_path, class_name)
        print(f"Loading {class_name} ...")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            except:
                continue
    return np.array(images), np.array(labels)


X_train, y_train = load_images(TRAIN_DIR)
X_test,  y_test  = load_images(TEST_DIR)

print(f"Train images: {X_train.shape[0]} | Test images: {X_test.shape[0]}")

# One-hot encoding
num_classes = len(class_folders)
Y_train_cat = to_categorical(y_train, num_classes)
Y_test_cat  = to_categorical(y_test, num_classes)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
history = model.fit(X_train, Y_train_cat, 
                    validation_data=(X_test, Y_test_cat),
                    epochs=35, 
                    batch_size=32, 
                    verbose=1)


model.save("isl_sign_language_model_36classes.h5")
print("\nModel successfully saved as 'isl_sign_language_model_36classes.h5'")

# Plot accuracy & loss
fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].plot(history.history['accuracy'], label='Train')
ax[0].plot(history.history['val_accuracy'], label='Validation')
ax[0].set_title('Accuracy Curve')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Train')
ax[1].plot(history.history['val_loss'], label='Validation')
ax[1].set_title('Loss Curve')
ax[1].legend()
plt.show()