print('MOBULE SCREEN DEFECT DETECTION MODEL USING CNN ARCHITECTURE AS WE ONLY NEED TO CLASSIFY')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.model_selection import train_test_split
import shutil

dataset_path = r"D:\Datasets\Mobile_screen_defects\MSD-US\test"  # Change this to your dataset path
split_base_dir = r"D:\Datasets\Mobile_screen_defects\MSD-US\split data"  # Directory to store split data

# Define class labels
class_labels = ["good", "oil", "scratch", "stain"]

# Define split ratios
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

# Create directories for split dataset
for split in ["train", "val", "test"]:
    split_path = os.path.join(split_base_dir, split)
    os.makedirs(split_path, exist_ok=True)
    for category in class_labels:
        os.makedirs(os.path.join(split_path, category), exist_ok=True)

# Split data into train, validation, and test sets as because we're going to use flow_from_fdirectory for better performance
for category in class_labels:
    category_path = os.path.join(dataset_path, category)
    images = os.listdir(category_path)
    
    train_imgs, test_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    val_imgs, test_imgs = train_test_split(test_imgs, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # Move images to respective folders
    for img in train_imgs:
        shutil.copy(os.path.join(category_path, img), os.path.join(split_base_dir, "train", category, img))
    
    for img in val_imgs:
        shutil.copy(os.path.join(category_path, img), os.path.join(split_base_dir, "val", category, img))

    for img in test_imgs:
        shutil.copy(os.path.join(category_path, img), os.path.join(split_base_dir, "test", category, img))


image_size = (128, 128)
batch_size = 32

# Augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=25,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest"
)

# No augmentation for validation and test data, only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load train, validation, and test data
train_generator = train_datagen.flow_from_directory(
    os.path.join(split_base_dir, "train"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(split_base_dir, "val"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(split_base_dir, "test"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)


# Model building

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),  # Dropout for overfitting prevention
    Dense(4, activation='softmax')
])


# Compiling the model
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Changed learning rate slightly
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

callbacks = [early_stopping]

epochs = 50

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks
)


test_loss, test_accuracy = model.evaluate(test_generator)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_true, y_pred, target_names=class_labels))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

import random

# Predicting images from test set, for observing model performance on new images. 
def plot_predictions(generator, model, num_images=20):
    x_test, y_test = next(generator)
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(4, 5, i+1)
        plt.imshow(x_test[i])
        plt.axis("off")
        actual_label = class_labels[np.argmax(y_test[i])]
        predicted_label = class_labels[y_pred[i]]
        color = "green" if actual_label == predicted_label else "red"
        plt.title(f"Actual: {actual_label}\nPred: {predicted_label}", color=color)
    plt.show()

plot_predictions(test_generator, model, num_images=20)


'''
accuracy: 0.9326 - loss: 0.1765

Test Accuracy: 0.9590
Test Loss: 0.1171


precision    recall  f1-score   support

        good       0.00      0.00      0.00         2
         oil       1.00      0.95      0.97        40
     scratch       0.95      0.97      0.96        40
       stain       0.93      1.00      0.96        40

    accuracy                           0.96       122
   macro avg       0.72      0.73      0.73       122
weighted avg       0.94      0.96      0.95       122


'''