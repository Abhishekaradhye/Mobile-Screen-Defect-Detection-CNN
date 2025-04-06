# Mobile Screen Defect Detection using Transfer Learning (MobileNetV2)

# Importing necessary libraries
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

dataset_path = r"D:\Datasets\Mobile_screen_defects\MSD-US\test"
split_base_dir = r"D:\Datasets\Mobile_screen_defects\MSD-US\split_data_transfer"
class_labels = ["good", "oil", "scratch", "stain"]

train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

# Parting folders for training, validation & test
for split in ["train", "val", "test"]:
    for category in class_labels:
        os.makedirs(os.path.join(split_base_dir, split, category), exist_ok=True)

for category in class_labels:
    category_path = os.path.join(dataset_path, category)
    images = os.listdir(category_path)
    train_imgs, test_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    val_imgs, test_imgs = train_test_split(test_imgs, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    for img in train_imgs:
        shutil.copy(os.path.join(category_path, img), os.path.join(split_base_dir, "train", category, img))
    for img in val_imgs:
        shutil.copy(os.path.join(category_path, img), os.path.join(split_base_dir, "val", category, img))
    for img in test_imgs:
        shutil.copy(os.path.join(category_path, img), os.path.join(split_base_dir, "test", category, img))

image_size = (224, 224)  # Required for MobileNetV2
batch_size = 32

# Image Data Generators
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
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Creating data generators
train_generator = train_datagen.flow_from_directory(
    os.path.join(split_base_dir, "train"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)
val_generator = val_test_datagen.flow_from_directory(
    os.path.join(split_base_dir, "val"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)
test_generator = val_test_datagen.flow_from_directory(
    os.path.join(split_base_dir, "test"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# Load MobileNetV2 base model (without top)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze all pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)  # Convert 3D to 1D
x = Dense(256, activation='relu')(x)  # New dense layer
x = Dropout(0.4)(x)  # Prevent overfitting
predictions = Dense(4, activation='softmax')(x)  # Output layer for 4 classes

# Combine base and head
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
callbacks = [early_stop, reduce_lr]

# Train the model
epochs = 40
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions and metrics
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot predictions on random test images
def plot_predictions(generator, model, class_labels, num_images=20):
    x_test, y_test = next(generator)
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(4, 5, i+1)
        plt.imshow(x_test[i])
        plt.axis("off")
        actual = class_labels[np.argmax(y_test[i])]
        predicted = class_labels[y_pred[i]]
        color = "green" if actual == predicted else "red"
        plt.title(f"Actual: {actual}\nPred: {predicted}", color=color)
    plt.tight_layout()
    plt.show()

plot_predictions(test_generator, model, class_labels)
