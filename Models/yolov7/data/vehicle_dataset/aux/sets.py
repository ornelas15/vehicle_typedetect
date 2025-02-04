import os

def create_image_list(dataset_dir):
    # Define paths for each set
    train_dir = os.path.join(dataset_dir, 'train', 'images')
    valid_dir = os.path.join(dataset_dir, 'valid', 'images')
    test_dir = os.path.join(dataset_dir, 'test', 'images')

    # Define file paths to store relative image paths
    train_txt = os.path.join(dataset_dir, 'train.txt')
    valid_txt = os.path.join(dataset_dir, 'valid.txt')
    test_txt = os.path.join(dataset_dir, 'test.txt')

    # Function to collect relative image paths
    def collect_image_paths(image_dir):
        image_paths = []
        for subdir, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):  # Adjust to your image extensions
                    relative_path = os.path.relpath(os.path.join(subdir, file), dataset_dir)
                    image_paths.append(relative_path.replace(os.sep, '/'))  # Ensure proper path format for YOLO
        return image_paths

    # Collect paths for each dataset split
    train_images = collect_image_paths(train_dir)
    valid_images = collect_image_paths(valid_dir)
    test_images = collect_image_paths(test_dir)

    # Write the paths to the respective text files
    def write_to_file(file_path, image_paths):
        with open(file_path, 'w') as f:
            for path in image_paths:
                f.write(path + '\n./')

    write_to_file(train_txt, train_images)
    write_to_file(valid_txt, valid_images)
    write_to_file(test_txt, test_images)

    print("train.txt, valid.txt, and test.txt have been created!")

# Set the path to your dataset folder
dataset_dir = './vehicle_dataset'
create_image_list(dataset_dir)
