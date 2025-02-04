import os

def remap_labels(input_folder, output_folder, class_mapping):
    """
    Remap class labels in YOLO label files.

    Parameters:
    - input_folder (str): Path to the folder containing the original .txt files.
    - output_folder (str): Path to save the updated .txt files.
    - class_mapping (dict): Mapping of original class IDs to new class IDs.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, "r") as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    original_class = int(parts[0])
                    if original_class in class_mapping:
                        new_class = class_mapping[original_class]
                        updated_line = " ".join([str(new_class)] + parts[1:])
                        updated_lines.append(updated_line)

            with open(output_path, "w") as file:
                file.write("\n".join(updated_lines))

# Dataset mappings: Car, Truck, MotorCycle, Van, Bus

# dataset1_mapping = {0: 3, 1: 4, 2: 0, 3: 2, 4: 1}  # Van, Bus, Car, MotorCycle, Truck
# dataset2_mapping = {0: 0, 1: 1}  # Car, Truck
# dataset3_mapping = {0: 0, 1: 2, 2: 4, 3: 1, 4: 2, 5: 3}  # Car, MotorCycle, Bus, Truck, MotorCycle, Van
# dataset4_mapping = {0: 4, 1: 0, 2: 2, 3: 1, 4: 3} #['bus', 'car', 'motorbike', 'truck', 'van']
dataset5_mapping = {1: 4, 2: 0, 4: 2, 6: 1, 7: 3} #['x', 'Bus', 'Car', 'x', 'Motorbike', 'x', 'Truck', 'Van']


# # Remap Labels:
# input_folder_dataset1 = "./dataset1/valid/labels"
# output_folder_dataset1 = "./dataset1/valid/updated_labels"
# remap_labels(input_folder_dataset1, output_folder_dataset1, dataset1_mapping)

# input_folder_dataset2 = "./dataset2/valid/labels"
# output_folder_dataset2 = "./dataset2/valid/updated_labels"
# remap_labels(input_folder_dataset2, output_folder_dataset2, dataset2_mapping)

# input_folder_dataset3 = "./dataset3/valid/labels"
# output_folder_dataset3 = "./dataset3/valid/updated_labels"
# remap_labels(input_folder_dataset3, output_folder_dataset3, dataset3_mapping)

# input_folder_dataset4 = "./dataset5/labels"
# output_folder_dataset4 = "./dataset5/updated_labels"
# remap_labels(input_folder_dataset5, output_folder_dataset5, dataset5_mapping)

input_folder_dataset5 = "./dataset5/labels"
output_folder_dataset5 = "./dataset5/updated_labels"
remap_labels(input_folder_dataset5, output_folder_dataset5, dataset5_mapping)