import os
import filecmp

def remove_duplicate_files(input_folder, output_folder, images_folder):
    """
    Removes duplicate images and labels based on bounding box content.

    Parameters:
    - input_folder (str): Path to the folder containing the original .txt files.
    - output_folder (str): Path to save the updated .txt files.
    - images_folder (str): Path to the folder containing the corresponding images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_files = set()
    txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    for filename in txt_files:
        input_path = os.path.join(input_folder, filename)
        image_path = os.path.join(images_folder, os.path.splitext(filename)[0] + ".jpg")

        is_duplicate = False

        for processed_file in processed_files:
            processed_path = os.path.join(input_folder, processed_file)

            # Compare file contents to detect duplicates
            if filecmp.cmp(input_path, processed_path, shallow=False):
                is_duplicate = True
                break

        if not is_duplicate:
            # Copy unique file to output folder
            processed_files.add(filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, "r") as file:
                lines = file.readlines()

            with open(output_path, "w") as file:
                file.write("".join(lines))
        else:
            # Remove the duplicate image
            if os.path.exists(image_path):
                os.remove(image_path)
            # Remove the duplicate label file
            if os.path.exists(input_path):
                os.remove(input_path)


input_folder_dataset1 = "./dataset1/train/labels"
output_folder_dataset1 = "./dataset1/train/cleaned_labels"
images_folder_dataset1 = "./dataset1/train/images"
remove_duplicate_files(input_folder_dataset1, output_folder_dataset1, images_folder_dataset1)

input_folder_dataset2 = "./dataset2/train/labels"
output_folder_dataset2 = "./dataset2/train/cleaned_labels"
images_folder_dataset2 = "./dataset2/train/images"
remove_duplicate_files(input_folder_dataset2, output_folder_dataset2, images_folder_dataset2)

input_folder_dataset3 = "./dataset3/train/labels"
output_folder_dataset3 = "./dataset3/train/cleaned_labels"
images_folder_dataset3 = "./dataset3/train/images"
remove_duplicate_files(input_folder_dataset3, output_folder_dataset3, images_folder_dataset3)
