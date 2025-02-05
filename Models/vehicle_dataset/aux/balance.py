import os
import shutil

# Paths to datasets
final_images_folder = './final_dataset/images'
final_labels_folder = './final_dataset/labels'
dataset6_images_folder = './dataset6/images'
dataset6_labels_folder = './dataset6/labels'

# Class names and their corresponding current counts in final_dataset
final_dataset_counts = {
    'Car': 2786,
    'Truck': 1864,
    'Motorcycle': 1770,
    'Van': 898,
    'Bus': 963
}

# Target number of images per class
target_count = 2700

# Function to move files
def move_files(class_name, final_images_folder, final_labels_folder, dataset6_images_folder, dataset6_labels_folder, final_count, target_count):
    moved_count = 0
    
    # Iterate through label files in dataset6
    for filename in os.listdir(dataset6_labels_folder):
        if filename.endswith('.txt'):
            label_path = os.path.join(dataset6_labels_folder, filename)
            
            # Check if the label file contains the class
            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()
            
            if any(line.startswith(str(class_name)) for line in lines):
                # Move label and corresponding image
                base_name = os.path.splitext(filename)[0]
                image_path = os.path.join(dataset6_images_folder, base_name + '.jpg')
                
                if os.path.exists(image_path):
                    # Move files to final_dataset
                    shutil.move(label_path, os.path.join(final_labels_folder, filename))
                    shutil.move(image_path, os.path.join(final_images_folder, base_name + '.jpg'))
                    
                    moved_count += 1
                    final_count += 1

                    # Stop if the target count is reached
                    if final_count >= target_count:
                        print(f"{class_name}: Reached target of {target_count}.")
                        return moved_count

    print(f"{class_name}: Unable to reach target. Moved {moved_count} items.")
    return moved_count

# Class mapping (update based on your class mapping in label files)
class_mapping = {
    'Car': 0,
    'Truck': 1,
    'Motorcycle': 2,
    'Van': 3,
    'Bus': 4
}

# Iterate over classes and move files
for class_name, current_count in final_dataset_counts.items():
    class_id = class_mapping[class_name]
    print(f"Processing class: {class_name}")
    
    moved = move_files(
        class_id, 
        final_images_folder, 
        final_labels_folder, 
        dataset6_images_folder, 
        dataset6_labels_folder, 
        current_count, 
        target_count
    )

    print(f"{class_name}: Moved {moved} files to final_dataset.")

print("File moving process completed.")
