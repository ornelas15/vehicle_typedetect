import os
from collections import Counter
import matplotlib.pyplot as plt

def count_classes(label_dir, class_names):
    """
    Counts the number of instances for each class in the dataset.

    Args:
        label_dir (str): Path to the directory containing label files.
        class_names (list): List of class names.

    Returns:
        dict: A dictionary with class names as keys and counts as values.
    """
    class_counts = Counter()

    # Iterate through all label files
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):  # Process only .txt files
            label_path = os.path.join(label_dir, label_file)

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:  # Skip malformed lines
                        continue

                    class_id = int(parts[0])  # Class ID is the first value
                    class_counts[class_id] += 1

    # Map class IDs to class names
    class_count_dict = {class_names[i]: class_counts[i] for i in range(len(class_names))}
    return class_count_dict

def plot_class_distribution(class_counts):
    """
    Plots a bar chart for the class distribution.

    Args:
        class_counts (dict): Dictionary with class names as keys and counts as values.
    """
    # Data for the chart
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Class Distribution', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Define your classes
    class_names = ['Car', 'Truck', 'Motorcycle', 'Van', 'Bus']
    class_names = ['Bicycle', 'Bus', 'Car', 'Jeep', 'Motorbike', 'Tricycle', 'Truck', 'Van']

    # Path to your labels directory
    label_directory = "./d5/labels"

    # Count classes
    counts = count_classes(label_directory, class_names)
    print(counts) 

    # Plot class distribution
    plot_class_distribution(counts)
