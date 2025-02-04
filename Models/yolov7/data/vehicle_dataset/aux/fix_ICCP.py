import os
from PIL import Image

def fix_iccp(input_path, output_path):
    """Remove incorrect ICC profile from the image."""
    with Image.open(input_path) as img:
        img.save(output_path, icc_profile=None)  # Removes incorrect sRGB profile

# Define dataset directory
dataset_dir = "./vehicle_dataset"

# Define all dataset splits
splits = ['train', 'valid', 'test']

def process_images(split):
    # Define input and output folders for images
    input_folder = os.path.join(dataset_dir, split, 'images')
    output_folder = os.path.join(dataset_dir, split, 'images_fixed')
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process all images in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            fix_iccp(input_path, output_path)

    print(f"ICC profiles fixed for all images in {split}.")

# Process images for train, valid, and test splits
for split in splits:
    process_images(split)