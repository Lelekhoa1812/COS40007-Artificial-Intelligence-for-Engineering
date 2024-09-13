# STUDIO ACTIVITY 2: Develop your custom CNN for image classification
import os
import numpy as np
import json
from PIL import Image
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

# Load images and labels for RGB images
def load_rgb_data(folder):
    images = []
    labels = []
    label_map = {}  # Map of class labels to indices
    image_files = set(f.split('.')[0] for f in os.listdir(folder) if f.endswith('.png'))
    label_files = set(f.split('.')[0] for f in os.listdir(folder) if f.endswith('.json'))
    
    common_files = image_files.intersection(label_files)
    
    for file_name in common_files:
        image_path = os.path.join(folder, file_name + '.png')
        json_path = os.path.join(folder, file_name + '.json')
        
        with open(json_path) as f:
            data = json.load(f)
        image = Image.open(image_path).convert('RGB')  # Load as RGB (instead of grayscale 'L')
        image = image.resize((28, 28))                 # Resize to match input shape
        image = (img_to_array(image) / 255.0) - 0.5    # Normalize and convert to array
        images.append(image)
        
        # Extract labels from the JSON file
        for shape in data['shapes']:
            label = shape['label']
            if label not in label_map:
                label_map[label] = len(label_map)      # Assign new label index
            labels.append(label_map[label])
        if not data['shapes']:
            print(f"No labels found in {json_path}")
    images = np.array(images)
    
    if not labels:
        print("No labels were found. Please check your label files.")
        return images, np.array([]), label_map
    labels = np.array(labels)
    
    # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes=len(label_map))
    return images, labels, label_map

# Load the training data (can change to log-labelled)
train_images, train_labels, label_map = load_rgb_data('log-labelled')
print("Number of labels / classes: ", len(label_map))

# Build the model 
num_filters = 8
filter_size = 3
pool_size = 1

model = Sequential([
    Input(shape=(28, 28, 3)),                         
    Conv2D(num_filters, filter_size, activation='relu'),
    MaxPooling2D(pool_size=pool_size),
    Conv2D(num_filters * 2, filter_size, activation='relu'),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax' if len(label_map) > 2 else 'sigmoid'),
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy' if len(label_map) > 2 else 'binary_crossentropy',
    metrics=['accuracy'],
)

# Train the model
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_split=0.2,
)

# Save the model weights
model.save_weights('log_cnn.weights.h5')

# Predictions
predictions = model.predict(train_images[:5])
print("Predicted labels:", np.argmax(predictions, axis=1))
print("-------------------------------------------------------------------")  # Splitter


# STUDIO ACTIVITY 3: Transfer Learning with RestNet for image classification
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers

# 1. Global constants
NUM_CLASSES = 1
CHANNELS = 3
IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']

NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100
BATCH_SIZE_TESTING = 1

# Define the architecture identical to Activity 1
def build_base_model(input_shape=(28, 28, 3), num_filters=8, filter_size=3, pool_size=1):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(num_filters, filter_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(num_filters * 2, filter_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
    ])
    return model

# Load the base model and load pre-trained weights
base_model = build_base_model()
base_model.load_weights('log_cnn.weights.h5')

# Build the transfer learning model
transfer_model = Sequential([
    base_model,  # Include the pre-trained model
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax' if NUM_CLASSES > 2 else 'sigmoid'),
])

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy' if NUM_CLASSES > 2 else 'binary_crossentropy',
    metrics=['accuracy'],
)

transfer_model.summary()

# Compile the model with SGD optimizer
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
transfer_model.compile(optimizer=sgd, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)

# 3. Prepare Keras Data Generators
data_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
train_generator = data_generator.flow_from_directory(
    'path/to/train_data',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size=BATCH_SIZE_TRAINING,
    class_mode='categorical'
)
validation_generator = data_generator.flow_from_directory(
    'path/to/validation_data',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size=BATCH_SIZE_VALIDATION,
    class_mode='categorical'
)

# 4. Train the Model with Early Stopping and Model Checkpointing
cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='auto')
fit_history = transfer_model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=STEPS_PER_EPOCH_VALIDATION,
    callbacks=[cb_checkpointer, cb_early_stopper]
)
# Load the best weights
transfer_model.load_weights("best_model.h5")

# 5. Visualize Training Metrics
plt.figure(1, figsize=(8, 8)) 
plt.subplot(221)  
plt.plot(fit_history.history['accuracy'])  
plt.plot(fit_history.history['val_accuracy'])  
plt.title('Model Accuracy')  
plt.ylabel('Accuracy')  
plt.xlabel('Epoch')  
plt.legend(['Train', 'Validation']) 

plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('Model Loss')  
plt.ylabel('Loss')  
plt.xlabel('Epoch')  
plt.legend(['Train', 'Validation']) 

plt.show()

# 6. Testing and Prediction
test_generator = data_generator.flow_from_directory(
    directory='path/to/test_data',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size=BATCH_SIZE_TESTING,
    class_mode=None,
    shuffle=False
)
# Reset the test generator
test_generator.reset()
# Perform prediction
predictions = transfer_model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_class_indices = np.argmax(predictions, axis=1)
# Plot some test images with predictions
TEST_DIR = 'path/to/test_data'
f, ax = plt.subplots(5, 5, figsize=(8, 8))

for i in range(25):
    imgBGR = cv2.imread(os.path.join(TEST_DIR, test_generator.filenames[i]))
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    predicted_class = "Class 1" if predicted_class_indices[i] else "Class 0"    
    
    ax[i//5, i%5].imshow(imgRGB)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Predicted: {}".format(predicted_class))
plt.show()

# Save the results
results_df = pd.DataFrame({
    'id': test_generator.filenames,
    'label': predicted_class_indices
})
results_df['id'] = results_df['id'].str.extract('(\d+)').astype(int)
results_df.sort_values(by='id', inplace=True)
results_df.to_csv('submission.csv', index=False)
results_df.head()
