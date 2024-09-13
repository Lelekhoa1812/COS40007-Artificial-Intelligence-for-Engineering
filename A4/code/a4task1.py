# TASK 1: Develop CNN and Resnet50
# ACTIVITY 2: Develop a Simple CNN Model
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Constants
IMAGE_SIZE = 28 
BATCH_SIZE = 32
EPOCHS = 10

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    'Corrosion',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'Corrosion',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    'Test Set Rust',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
# Save the model
model.save('rust_cnn_model.h5')

# Predict on the first 5 test images
predictions = model.predict(test_generator)
print("Predicted labels (first 5):", np.round(predictions[:5]).astype(int).flatten())
print("-------------------------------------------------------------------")  # Splitter


# ACTIVITY 3: Develop a ResNet50 Model
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    'Corrosion',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'Corrosion',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    'Test Set Rust',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ResNet50 Model
resnet = ResNet50(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), pooling='avg', weights='imagenet')
resnet.trainable = False  # Freeze ResNet layers

model = Sequential([
    resnet,
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3),
    ModelCheckpoint('best_rust_resnet50_model.keras', monitor='val_loss', save_best_only=True)
]

# Train the model
fit_history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
# Save the model
model.save('rust_resnet50_model.h5')
# Model Summary
model.summary()

# Plotting Model Accuracy and Loss
plt.figure(1, figsize=(8, 8))
plt.subplot(221)
plt.plot(fit_history.history['accuracy'])
plt.plot(fit_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplot(222)
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Display Predicted Rust and No Rust Images
f, ax = plt.subplots(5, 4, figsize=(8, 8))
for i in range(20):
    img = plt.imread(test_generator.filepaths[i])
    ax[i//4, i%4].imshow(img)
    ax[i//4, i%4].axis('off')
    predicted_class = "Rust" if predictions[i] >= 0.5 else "No Rust"
    ax[i//4, i%4].set_title(f"Predicted: {predicted_class}")
plt.show()
print("-------------------------------------------------------------------")  # Splitter