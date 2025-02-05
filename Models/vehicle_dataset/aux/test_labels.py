import os
import cv2

def process_labels(label_path, image):
    """
    Process label file and draw bounding boxes on the image before rescaling.

    Args:
        label_path (str): Path to the label file.
        image (ndarray): Image to draw bounding boxes on.
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        
        # Check if there are exactly 5 parts in the line (class_id and 4 bbox values)
        if len(parts) != 5:
            print(f"Skipping malformed line: {line.strip()}")
            continue

        cls, x_center, y_center, width, height = map(float, parts)

        # Convert from normalized YOLO coordinates to pixel values
        h, w, _ = image.shape
        x_center_pixel = int(x_center * w)
        y_center_pixel = int(y_center * h)
        width_pixel = int(width * w)
        height_pixel = int(height * h)

        # Calculate the bounding box's top-left and bottom-right corners
        x_min = x_center_pixel - width_pixel // 2
        y_min = y_center_pixel - height_pixel // 2
        x_max = x_center_pixel + width_pixel // 2
        y_max = y_center_pixel + height_pixel // 2

        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Print the raw bounding box values before rescaling
        print(f"Before rescaling: class={cls}, x_center={x_center}, y_center={y_center}, width={width}, height={height}")
        print(f"Bounding Box (pixels): x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

def check_all_images(base_dir):
    """
    Check all images in the dataset folder by displaying their bounding boxes before rescaling.
    Allows user to press Enter to move to the next image.

    Args:
        base_dir (str): Path to the base dataset directory.
    """
    subsets = ['train_resized', 'valid_resized', 'test_resized']
    for subset in subsets:
        image_dir = os.path.join(base_dir, subset, 'images')
        label_dir = os.path.join(base_dir, subset, 'labels')

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

            # Show the image with the bounding boxes before rescaling
            print(f"Inspecting image: {image_name}")
            process_labels(label_path, image)

            # Display the image with the bounding boxes
            cv2.imshow("Image with Bounding Boxes", image)
            
            # Wait for the user to press Enter to proceed
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()  # Close the image window

if __name__ == "__main__":
    dataset_dir = "vehicle_dataset"  # Provide the correct path to your dataset directory
    check_all_images(dataset_dir)
