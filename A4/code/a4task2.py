# TASK 2: Develop Mask RCNN for detecting log
# STEP 0: Randomly take 10 images for testing
import os
import random
import shutil
# Define the directories
log_labelled_dir = "log-labelled"
test_set_log_dir = "Test Set Log"
# Ensure the test set folder exists
os.makedirs(test_set_log_dir, exist_ok=True)
# Get all PNG files from the log-labelled directory
png_files = [f for f in os.listdir(log_labelled_dir) if f.endswith('.png')]
# Randomly select 10 PNG files
selected_files = random.sample(png_files, 10)
# Move selected PNG files and their corresponding JSON files
for png_file in selected_files:
    # Construct the file paths
    png_path = os.path.join(log_labelled_dir, png_file)
    json_path = os.path.splitext(png_path)[0] + '.json'
    # Move the PNG file
    shutil.move(png_path, os.path.join(test_set_log_dir, png_file))
    # Check if the corresponding JSON file exists and move it
    if os.path.exists(json_path):
        shutil.move(json_path, os.path.join(test_set_log_dir, os.path.basename(json_path)))
print("10 PNG and corresponding JSON files have been moved to the Test Set Log folder.")

# STEP 1: Convert LabelMe Annotations to COCO Format
import os
import json
import numpy as np
import cv2
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torch.utils.data import Dataset
from engine import train_one_epoch
import utils
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

labelme_folder = "log-labelled"
export_dir = "export/coco"
# You already converted annotations, so this step is commented out
# labelme2coco.convert(labelme_folder, export_dir)
print("Step 1 completed -------------------------------------------------------------------")

# STEP 2: Define the Dataset Class
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
        if len(masks) > 0:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)  # Handle images with no logs
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
print("Step 2 completed -------------------------------------------------------------------")

# STEP 3: Create the Mask R-CNN Model
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model
print("Step 3 completed -------------------------------------------------------------------")

# STEP 4: Training Loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Define transformations
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
# Load datasets
dataset = LogDataset('log-labelled', get_transform(train=True))
dataset_test = LogDataset('Test Set Log', get_transform(train=False))
subset_size = 20  # Use only 20 images for faster training, reduce runtime
dataset = torch.utils.data.Subset(dataset, list(range(subset_size))) # Reduce subset_size to reduce runtime
# Create data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, collate_fn=utils.collate_fn)
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
num_epochs = 1
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
print("Training complete!")
print("Step 4 completed -------------------------------------------------------------------")

# STEP 5: Testing and Counting Logs with Visualization
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
    # Save and display the image
    output_dir = "Log Detection"  # Ensure directory for output images exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output_image_{i}.png")
    cv2.imwrite(output_path, image_np)
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
