import os
import cv2

def resize_and_convert(image, label_path, target_size=(640, 640)):
    """
    Resize the image to the target size and adjust bounding boxes.
    
    Args:
        image (ndarray): The input image to resize.
        label_path (str): Path to the label file for the image.
        target_size (tuple): The target size to resize the image to, e.g., (640, 640).
        
    Returns:
        resized_image (ndarray): Resized image.
        new_labels (list): List of new bounding boxes in YOLO format after resizing.
    """
    # Read the label file and extract bounding boxes
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    new_labels = []
    original_height, original_width, _ = image.shape

    # Calculate the scaling factors
    scale_x = target_size[0] / original_width
    scale_y = target_size[1] / original_height

    # Process each bounding box in the label file
    for line in lines:
        parts = line.strip().split()
        
        # Check if the line is well-formed (class_id and 4 bbox values)
        if len(parts) != 5:
            print(f"Skipping malformed line: {line.strip()}")
            continue

        cls, x_center, y_center, width, height = map(float, parts)

        # Convert from YOLO to pixel coordinates
        x_center_pixel = x_center * original_width
        y_center_pixel = y_center * original_height
        width_pixel = width * original_width
        height_pixel = height * original_height

        # Resize the bounding box based on the scaling factors
        x_center_resized = x_center_pixel * scale_x
        y_center_resized = y_center_pixel * scale_y
        width_resized = width_pixel * scale_x
        height_resized = height_pixel * scale_y

        # Convert back to YOLO format (normalized coordinates in the range [0, 1])
        x_center_yolo = x_center_resized / target_size[0]
        y_center_yolo = y_center_resized / target_size[1]
        width_yolo = width_resized / target_size[0]
        height_yolo = height_resized / target_size[1]

        # Append the new bounding box to the list
        new_labels.append(f"{cls} {x_center_yolo} {y_center_yolo} {width_yolo} {height_yolo}")
    
    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)
    
    return resized_image, new_labels

def check_and_resize_images(base_dir, target_size=(640, 640)):
    """
    Resize all images in the dataset, adjust bounding boxes, and save them as .jpg files.
    Saves the resized images and updated label files in a new folder.
    
    Args:
        base_dir (str): Path to the base dataset directory.
        target_size (tuple): The target size to resize the images to.
    """
    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        image_dir = os.path.join(base_dir, subset, 'images')
        label_dir = os.path.join(base_dir, subset, 'labels')

        # Create the new directories for resized images and labels
        new_image_dir = os.path.join(base_dir, f"{subset}_resized", 'images')
        new_label_dir = os.path.join(base_dir, f"{subset}_resized", 'labels')
        
        # Create the directories if they do not exist
        os.makedirs(new_image_dir, exist_ok=True)
        os.makedirs(new_label_dir, exist_ok=True)

        for image_name in os.listdir(image_dir):
            if not image_name.lower().endswith(('png', 'jpg', 'jpeg')):
                continue

            # Paths for image and label
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Resize image and update bounding boxes
            resized_image, new_labels = resize_and_convert(image, label_path, target_size)

            # Save the resized image as .jpg in the new directory
            resized_image_name = os.path.splitext(image_name)[0] + '.jpg'
            resized_image_path = os.path.join(new_image_dir, resized_image_name)
            cv2.imwrite(resized_image_path, resized_image)

            # Save the new labels to a new file in the new directory
            resized_label_name = os.path.splitext(image_name)[0] + '.txt'
            resized_label_path = os.path.join(new_label_dir, resized_label_name)
            with open(resized_label_path, 'w') as f:
                for label in new_labels:
                    f.write(label + "\n")

            print(f"Processed {image_name} and saved resized image as {resized_image_name} in the new folder.")

if __name__ == "__main__":
    dataset_dir = "vehicle_dataset"  # Provide the correct path to your dataset directory
    check_and_resize_images(dataset_dir)
