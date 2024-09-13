# TASK 3: Extending log labelling to another class
import os
import json

# Define the directory containing the PNG and JSON files
log_images_dir = "log-images"

# Function to update the label of broken logs to "detected_log"
def update_labels(json_file_path):
    # Open and read the JSON file
    with open(json_file_path, 'r') as f:
        annotation = json.load(f)
    
    # Iterate over all shapes in the JSON (representing different annotations)
    for shape in annotation['shapes']:
        # Check if the label is 'broken' or needs to be changed
        if shape['label'] == 'broken':
            shape['label'] = 'detected_log'  # Update the label
    
    # Save the updated annotation back to the same JSON file
    with open(json_file_path, 'w') as f:
        json.dump(annotation, f, indent=4)
    
    print(f"Updated labels in {json_file_path}")

# Get all JSON files from the log-images directory
json_files = [f for f in os.listdir(log_images_dir) if f.endswith('.json')]

# Update labels in each JSON file
for json_file in json_files:
    json_file_path = os.path.join(log_images_dir, json_file)
    update_labels(json_file_path)
print("All labels updated successfully.")
