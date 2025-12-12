import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from extra_keras_datasets import emnist


np.random.seed(42)
tf.random.set_seed(42)


manual_zip_path = "D:/mini/matlab.zip"

if not os.path.exists(manual_zip_path):
    raise FileNotFoundError(f"EMNIST zip not found at: {manual_zip_path}")


print("Loading EMNIST digits dataset from manual zip...")
(X_train, y_train), (X_test, y_test) = emnist.load_data(type='digits', path=manual_zip_path)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")


X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


y_train_encoded = keras.utils.to_categorical(y_train, num_classes)
y_test_encoded = keras.utils.to_categorical(y_test, num_classes)


X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
    X_train, y_train_encoded, test_size=0.1, random_state=42, stratify=y_train
)


train_datagen = ImageDataGenerator(
    rotation_range=10, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, fill_mode='nearest'
)
val_datagen = ImageDataGenerator()

batch_size = 128
train_generator = train_datagen.flow(X_train, y_train_encoded, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val_encoded, batch_size=batch_size)


model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()


model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',  
    metrics=['accuracy']
)


os.makedirs("models", exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint("models/best_model.h5", save_best_only=True, monitor="val_accuracy", verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1)
]


print("🚀 Starting full training on complete EMNIST dataset...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    steps_per_epoch=len(X_train) // batch_size,
    validation_steps=len(X_val) // batch_size,
    callbacks=callbacks,
    verbose=1
)

best_model = keras.models.load_model("models/best_model.h5")
test_loss, test_accuracy = best_model.evaluate(X_test, y_test_encoded, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")


best_model.save("models/alphanumeric_model.keras")



class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
}

with open("models/class_mapping.json", "w") as f:
    json.dump(class_mapping, f, indent=2)

print("Training complete! Models and mappings saved to 'models/'")
