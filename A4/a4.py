# # TASK 1: Develop CNN and Resnet50
# # ACTIVITY 2: Develop a Simple CNN Model
# import os
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import to_categorical

# # Constants
# IMAGE_SIZE = 28 
# BATCH_SIZE = 32
# EPOCHS = 10

# # Data Generators
# train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
# test_datagen = ImageDataGenerator(rescale=1.0/255)
# train_generator = train_datagen.flow_from_directory(
#     'Corrosion',
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='training'
# )

# validation_generator = train_datagen.flow_from_directory(
#     'Corrosion',
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='validation'
# )

# test_generator = test_datagen.flow_from_directory(
#     'Test Set Rust',
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='binary'
# )

# # CNN Model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE,
#     epochs=EPOCHS
# )

# # Evaluate on the test set
# test_loss, test_acc = model.evaluate(test_generator)
# print('Test accuracy:', test_acc)
# # Save the model
# model.save('rust_cnn_model.h5')

# # Predict on the first 5 test images
# predictions = model.predict(test_generator)
# print("Predicted labels (first 5):", np.round(predictions[:5]).astype(int).flatten())
# print("-------------------------------------------------------------------")  # Splitter


# # ACTIVITY 3: Develop a ResNet50 Model
# import numpy as np
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import matplotlib.pyplot as plt

# # Constants
# IMAGE_SIZE = 224
# BATCH_SIZE = 32
# EPOCHS = 10

# # Data Generators
# train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
# test_datagen = ImageDataGenerator(rescale=1.0/255)

# train_generator = train_datagen.flow_from_directory(
#     'Corrosion',
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='training'
# )

# validation_generator = train_datagen.flow_from_directory(
#     'Corrosion',
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='validation'
# )

# test_generator = test_datagen.flow_from_directory(
#     'Test Set',
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='binary'
# )

# # ResNet50 Model
# resnet = ResNet50(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), pooling='avg', weights='imagenet')
# resnet.trainable = False  # Freeze ResNet layers

# model = Sequential([
#     resnet,
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Callbacks
# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=3),
#     ModelCheckpoint('best_rust_resnet50_model.keras', monitor='val_loss', save_best_only=True)
# ]

# # Train the model
# fit_history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE,
#     epochs=EPOCHS,
#     callbacks=callbacks
# )

# # Evaluate on the test set
# test_loss, test_acc = model.evaluate(test_generator)
# print('Test accuracy:', test_acc)
# # Save the model
# model.save('rust_resnet50_model.h5')
# # Model Summary
# model.summary()

# # Plotting Model Accuracy and Loss
# plt.figure(1, figsize=(8, 8))
# plt.subplot(221)
# plt.plot(fit_history.history['accuracy'])
# plt.plot(fit_history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.subplot(222)
# plt.plot(fit_history.history['loss'])
# plt.plot(fit_history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# # Display Predicted Rust and No Rust Images
# f, ax = plt.subplots(5, 4, figsize=(8, 8))
# for i in range(20):
#     img = plt.imread(test_generator.filepaths[i])
#     ax[i//4, i%4].imshow(img)
#     ax[i//4, i%4].axis('off')
#     predicted_class = "Rust" if predictions[i] >= 0.5 else "No Rust"
#     ax[i//4, i%4].set_title(f"Predicted: {predicted_class}")
# plt.show()
# print("-------------------------------------------------------------------")  # Splitter


# TASK 2: Develop Mask RCNN for detecting log
import labelme2coco
import os
import json
import numpy as np
import cv2
from mrcnn import utils
import mrcnn.model 
import mrcnn.config
import mrcnn.visualize
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# STEP 1: Convert LabelMe Annotations to COCO Format
labelme_folder = "log-labelled"
export_dir = "export/coco"
labelme2coco.convert(labelme_folder, export_dir) # Convert LabelMe annotations from the training folder, export to export/coco/dataset.json

train_dir = "log-labelled"
test_dir = "Test Set Log"
print("Step 1 completed -------------------------------------------------------------------")  # Splitter

# # STEP 2: Mask R-CNN Model Development
# class LogConfig(Config):
#     NAME = "log_cfg"
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     NUM_CLASSES = 1 + 1  # background + log
#     STEPS_PER_EPOCH = 100

# class LogDataset(utils.Dataset):
#     def load_log(self, dataset_dir, subset):
#         self.add_class("log", 1, "log")
#         image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
#         for image_file in image_files:
#             json_file = os.path.splitext(image_file)[0] + '.json'
#             json_path = os.path.join(dataset_dir, json_file)
#             if os.path.exists(json_path):
#                 with open(json_path) as f:
#                     annotations = json.load(f)
#                 image_path = os.path.join(dataset_dir, image_file)
#                 height, width = cv2.imread(image_path).shape[:2]
#                 self.add_image(
#                     "log",
#                     image_id=image_file,
#                     path=image_path,
#                     width=width,
#                     height=height,
#                     annotations=annotations['shapes']
#                 )

#     def load_mask(self, image_id):
#         info = self.image_info[image_id]
#         annotations = info['annotations']
#         masks = np.zeros([info['height'], info['width'], len(annotations)], dtype=np.uint8)
#         class_ids = []
#         for i, annotation in enumerate(annotations):
#             if annotation['label'] == "log":
#                 class_ids.append(self.class_names.index("log"))
#                 masks[:, :, i] = self.ann_to_mask(annotation, info['height'], info['width'])
#         return masks, np.array(class_ids, dtype=np.int32)

#     def ann_to_mask(self, ann, height, width):
#         mask = np.zeros((height, width), dtype=np.uint8)
#         poly = np.array(ann['points']).reshape((-1, 2)).astype(np.int32)
#         cv2.fillPoly(mask, [poly], 1)
#         return mask

# # 3. Train the Mask R-CNN Model
# dataset_train = LogDataset()
# dataset_train.load_log(export_dir, "dataset")
# dataset_train.prepare()

# dataset_val = LogDataset()
# dataset_val.load_log(test_dir, "dataset")
# dataset_val.prepare()

# config = LogConfig()

# # Test logs folder exist
# log_dir = os.path.join(os.getcwd(), "logs")
# if not os.path.exists(log_dir):
#     print("logs folder not exist hence create new!")
#     os.makedirs(log_dir)
# model = MaskRCNN(mode="training", config=config, model_dir=log_dir)
# model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# model.set_log_dir(log_dir)  # Ensure that the log directory is set correctly
# model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')
# model.keras_model.summary()

# # 4. Save the trained model weights
# model_path = "mask_rcnn_log.h5"
# model.keras_model.save_weights(model_path)

# # STEP 3: Test the Model and Generate Output Images
# model = MaskRCNN(mode="inference", config=config, model_dir="logs")
# model.load_weights(model_path, by_name=True)

# for image_file in os.listdir(test_dir):
#     if image_file.endswith('.png'):
#         image_path = os.path.join(test_dir, image_file)
#         image = cv2.imread(image_path)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         r = model.detect([image_rgb], verbose=1)[0]
#         mrcnn.visualize.display_instances(image=image_rgb, 
#                                           boxes=r['rois'], 
#                                           masks=r['masks'], 
#                                           class_ids=r['class_ids'], 
#                                           class_names=["BG", "log"], 
#                                           scores=r['scores'])
#         num_logs = len(r['rois'])
#         print(f'Number of detected logs in {image_file}: {num_logs}')

# STEP 2: Define the Dataset Class
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torch.utils.data import Dataset

class LogDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted([f for f in os.listdir(root) if f.endswith('.png')])
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        json_path = os.path.splitext(img_path)[0] + '.json'
        # Load image
        img = read_image(img_path).float() / 255.0  # Normalize image to range [0, 1]
        # Load annotation from JSON
        with open(json_path) as f:
            annotations = json.load(f)
        # Extract the masks from the annotation
        masks = []
        height, width = img.shape[1], img.shape[2]
        for shape in annotations['shapes']:
            if shape['label'] == 'log':
                mask = np.zeros((height, width), dtype=np.uint8)
                poly = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [poly], 1)
                masks.append(torch.tensor(mask, dtype=torch.uint8))
        masks = torch.stack(masks, dim=0)
        # Compute bounding boxes from masks
        boxes = masks_to_boxes(masks)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Assuming all logs
        # Create the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target
    
    def __len__(self):
        return len(self.imgs)
print("Step 2 completed -------------------------------------------------------------------")  # Splitter

# STEP 3: Create the Mask R-CNN Model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model
print("Step 3 completed -------------------------------------------------------------------")  # Splitter

# STEP 4: Training Loop
import torch
from engine import train_one_epoch, evaluate
import utils
from torchvision.transforms import v2 as T

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Define transformations
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
# Load dataset
dataset = LogDataset('log-labelled', get_transform(train=True))
dataset_test = LogDataset('Test Set Log', get_transform(train=False))
# Split dataset
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-10])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])
# Create data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
# Get the model
model = get_model_instance_segmentation(num_classes=2)  # 1 class (log) + background
model.to(device)
# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)
print("Training complete!")
print("Step 4 completed -------------------------------------------------------------------")  # Splitter

# STEP 5: Testing and Counting Logs with Visualization
import cv2
import matplotlib.pyplot as plt

model.eval()
for i, (image, _) in enumerate(dataset_test):
    with torch.no_grad():
        prediction = model([image.to(device)])
    # Extract information from predictions
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    masks = prediction[0]['masks'].cpu().numpy()
    num_logs = len(boxes)
    print(f"Detected {num_logs} logs in the image.")
    # Convert image to numpy array for OpenCV
    image_np = image.mul(255).permute(1, 2, 0).byte().numpy()
    # Draw bounding boxes and labels on the image
    for j in range(num_logs):
        box = boxes[j].astype(int)
        score = scores[j]
        mask = masks[j, 0]
        # Draw bounding box
        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # Draw confidence score
        label = f"{score:.2f}"
        cv2.putText(image_np, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Apply mask
        image_np[mask > 0.5] = [0, 0, 255]  # Red color for the mask
    # Save or display the image
    output_path = f"output_image_{i}.png"
    cv2.imwrite(output_path, image_np)
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
