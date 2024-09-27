# STEP 1: Convert Given Annotation Format to YOLO Format
# YOLO format: <object_class> <x_center> <y_center> <width> <height>
import os
import pandas as pd

def convert_csv_to_yolo(csv_file, images_dir, yolo_labels_dir):
    if not os.path.exists(yolo_labels_dir):
        os.makedirs(yolo_labels_dir)

    # Load the CSV file
    df = pd.read_csv(csv_file)

    for image_file in df['filename'].unique():
        image_annotations = df[df['filename'] == image_file]
        image_path = os.path.join(images_dir, image_file)

        img_width = image_annotations.iloc[0]['width']
        img_height = image_annotations.iloc[0]['height']

        # Create YOLO label file
        yolo_label_file = os.path.join(yolo_labels_dir, os.path.splitext(image_file)[0] + '.txt')

        with open(yolo_label_file, 'w') as f:
            for _, row in image_annotations.iterrows():
                # Convert VOC to YOLO format
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                bbox_width = (xmax - xmin) / img_width
                bbox_height = (ymax - ymin) / img_height

                # Write the YOLO format
                f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                
        print(f"Converted {image_file} to YOLO format.")

# Directory set ups
train_csv = "annotations/train_labels.csv"
test_csv = "annotations/test_labels.csv"
train_images_dir = "images/train"
test_images_dir = "images/test"
yolo_train_labels_dir = "labels/train"
yolo_test_labels_dir = "labels/test"

# Convert train and test annotations
convert_csv_to_yolo(train_csv, train_images_dir, yolo_train_labels_dir)
convert_csv_to_yolo(test_csv, test_images_dir, yolo_test_labels_dir)
print("------------------------------------------------------------------------------") # Splitter

# STEP 2: Train a YOLO Model with 400 Images
import random
import shutil
import os
import pandas as pd
import subprocess

# Directory set ups
train_source_dir = "images/train"  
train_target_dir = "selected_train_images/images"

# Function to select random images from source_dir and copy them to target_dir
def select_random_images(source_dir, target_dir, num_images=400):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists("selected_train_images/labels"): # Target dir for annotations
        os.makedirs("selected_train_images/labels")

    images = [f for f in os.listdir(source_dir) if f.endswith('.jpg') or f.endswith('.JPG')]
    
    # Check if the number of available images is less than the requested number
    if len(images) < num_images:
        print(f"Only {len(images)} images available, selecting all of them.")
        num_images = len(images)
    selected_images = random.sample(images, num_images)

    for image in selected_images:
        # Copy the image to the target directory
        shutil.copy(os.path.join(source_dir, image), os.path.join(target_dir, image))
        
        # Check for the corresponding annotation file (YOLO format .txt)
        annotation_file = os.path.splitext(image)[0] + ".txt"
        annotation_src_path = os.path.join("labels/train", annotation_file)                 # annotations src dir
        annotation_dst_path = os.path.join("selected_train_images/labels", annotation_file) # annotations dst dir
        
        if os.path.exists(annotation_src_path):
            shutil.copy(annotation_src_path, annotation_dst_path)
        else:
            print(f"Warning: Annotation file not found for {image}. Skipping {annotation_file}.")
    print(f"Copied {num_images} images to {target_dir} and {num_images} annotations to selected_train_images/labels")

# Call the function to randomly select 400 images
select_random_images(train_source_dir, train_target_dir)

# Bash script to install yolov5m.pt
# wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt

# Usage with bash command
# Small pre-trained model usage
# Script to train data using YOLOv5 train.py model, with 400 images, image size 640x640, 10 epochs and utilising the pre-trained model weights from yolov5s.pt
# python3 train.py --img 640 --batch 16 --epochs 10 --data data.yaml --weights yolov5s.pt --cache
# Medium pre-trained model usage
# Script to train data using YOLOv5 train.py model, with 400 images, image size 640x640, 50 epochs and utilising the pre-trained model weights from yolov5m.pt
# python3 train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5m.pt --cache

# Usage with jupyter notebook command / google colab
# Script to train data using YOLOv5 train.py model, with 400 images, image size 640x640, 50 epochs and utilising the pre-trained model weights from yolov5m.pt
# !python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5m.pt --cache

# Trained model will be saved at runs/train/exp/weights
print("------------------------------------------------------------------------------") # Splitter

# STEP 3: Compute IoU for Test Data and Evaluations
import torch
import cv2
import pandas as pd
import random
import os

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')  # Using pretrained model from step 2

# Compute the IoU result
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Load YOLO ground truth annotations (test txt files)
def load_yolo_annotation(yolo_label_file, img_width, img_height):
    with open(yolo_label_file, 'r') as file:
        boxes = []
        for line in file.readlines():
            _, x_center, y_center, width, height = map(float, line.strip().split())
            xmin = (x_center - width / 2) * img_width
            ymin = (y_center - height / 2) * img_height
            xmax = (x_center + width / 2) * img_width
            ymax = (y_center + height / 2) * img_height
            boxes.append([xmin, ymin, xmax, ymax])
    return boxes

# Evaluate the model on 40 randomly sampled test images
def evaluate_test_images(test_dir, yolo_labels_dir, output_csv, model):
    images = [f for f in os.listdir(test_dir) if f.endswith('.jpg') or f.endswith('.JPG')]
    results = []

    # Convert the unique filenames to a list and randomly sample 40 images
    test_images = random.sample(images, 40)  # Randomly sample 40 test images
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        # Perform inference using the model
        results_inference = model(img)
        pred_boxes = results_inference.xyxy[0].cpu().numpy()[:, :4]  # Get the predicted boxes
        pred_scores = results_inference.xyxy[0].cpu().numpy()[:, 4]  # Get the confidence scores

        # Get the YOLO ground truth for the image from labels/test
        yolo_label_file = os.path.join(yolo_labels_dir, os.path.splitext(img_name)[0] + '.txt')
        if os.path.exists(yolo_label_file):
            gt_boxes = load_yolo_annotation(yolo_label_file, img_width, img_height)
        else:
            gt_boxes = []  # No annotations if the label file doesn't exist

        # Calculate IoU for each prediction
        if len(pred_boxes) == 0:  # No detections
            iou = 0
            confidence = 0
        else:
            ious = []
            confidences = []
            for pred_box, confidence in zip(pred_boxes, pred_scores):
                if gt_boxes:  # Only compute IoU if ground-truth boxes are available
                    ious.append(max([compute_iou(pred_box, gt_box) for gt_box in gt_boxes]))
                else:
                    ious.append(0)  # If no ground truth, IoU is 0
                confidences.append(confidence)

            iou = max(ious)                # Take the maximum IoU
            confidence = max(confidences)  # Take the maximum confidence
        # Append the result
        results.append([img_name, confidence, iou])

    # Save results to CSV
    output_df = pd.DataFrame(results, columns=['image_name', 'confidence_value', 'iou_value'])
    output_df.to_csv(output_csv, index=False)

# # Example usage:
# evaluate_test_images("images/test", "annotations/test_labels.csv", "iou_results.csv", model)

# Iterative training and testing until 80% IoU > 90%
def iterative_training_and_testing():
    threshold_met = False
    iteration = 1

    while not threshold_met:
        print(f"Iteration {iteration}: Training and Testing")

        # Step a: Call the script to retrain the model using a new set of training images
        # Make sure the weights of the last model are used for training the next one.
        # Training with 400 images, image size 640x640, batch size 32 (improve generalization), 20 epochs
        # Trained model will be saved at runs/train/exp{n+1}/weights with n to be the iteration ID ranging from exp2
        os.system("python3 train.py --img 640 --batch 32 --epochs 20 --data data.yaml --weights runs/train/exp/weights/best.pt --cache")

        # Step b: Call the evaluation function for 40 test images
        evaluate_test_images("images/test", "labels/test", f"iou_results_iter_{iteration}.csv", model)

        # Step c: Check if 80% of the images have an IoU value > 90%
        iou_results = pd.read_csv(f"iou_results_iter_{iteration}.csv")
        iou_over_90 = (iou_results['iou_value'] > 0.9).sum()
        total_images = len(iou_results)
        percentage_iou_over_90 = (iou_over_90 / total_images) * 100

        print(f"Iteration {iteration}: {percentage_iou_over_90}% images have IoU > 90%")

        # Step d: If 80% of the images have IoU > 90%, stop training
        if percentage_iou_over_90 >= 80:
            iou_results.to_csv("final_model.csv", index=False) # save current model as final model
            print("Threshold met. Stopping further iterations.")
            threshold_met = True
        else:
            print("Threshold not met. Continue the iterations.")
            iteration += 1

# Start the iterative training and testing process
iterative_training_and_testing()

# STEP 4: Real-Time Graffiti Detection on Video
import torch
import cv2
import os
import pandas as pd

# Load the final model from iteration 1 as we determined this to be the final model 
model_path = 'runs/train/exp2/weights/best.pt'        # Final model path (best.pt from iteration 1)
# model_path = 'runs/train/exp{i+1}/weights/best.pt'  # Custom final model path with different iteration ID 'i' 
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

video_dir = 'live_video'      # Directory containing video files
output_dir = 'output_videos'  # Directory to save output videos with graffiti detections

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to detect graffiti in video and save output
def detect_graffiti_in_video(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video frame width, height, and frames per second (fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define video writer to save output video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference on the current frame
        results = model(frame)
        # Draw bounding boxes and labels on the frame
        results.render()
        frame_with_detections = results.ims[0]
        # Write the frame with detections to the output video
        out.write(frame_with_detections)
        
        # Display the frame (optional for real-time visualization)
        cv2.imshow('Graffiti Detection', frame_with_detections)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Process each video in the live_video directory
for video_file in os.listdir(video_dir):
    if video_file.endswith(('.mp4', '.avi', '.mov')): # Handling different video extension formats (although we only use mp4 this task)
        video_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, f"output_{video_file}")
        print(f"Processing {video_file}...")
        detect_graffiti_in_video(video_path, output_path)
        print(f"Finished processing {video_file}. Saved to {output_path}")

print("All videos processed. Graffiti detection complete.")
