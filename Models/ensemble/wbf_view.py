import os
from ensemble_boxes import weighted_boxes_fusion
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def run_yolov7(weights, source, conf, img_size, output_dir, model_name):
    """Run YOLOv7 inference and save results."""
    absolute_output_dir = os.path.abspath(output_dir)
    command = (
        f"cd ./yolov7 && "
        f"python detect.py --weights {weights} --source {source} --conf-thres {conf} --img-size {img_size} "
        f"--project {absolute_output_dir} --name {model_name} --exist-ok --save-txt --save-conf && cd .."
    )
    print(f"Running YOLOv7 command: {command}")
    os.system(command)
    print(f"Results saved for YOLOv7 in {absolute_output_dir}/{model_name}")


def run_yolov9(weights, source, conf, img_size, output_dir, model_name):
    """Run YOLOv9 inference directly."""
    yolov9_dir = "./yolov9"
    command = (
        f"python {os.path.join(yolov9_dir, 'detect_dual.py')} "
        f"--weights {weights} --source {source} --conf-thres {conf} --img-size {img_size} "
        f"--project {output_dir} --name {model_name} --exist-ok --save-txt --save-conf"
    )
    print(f"Running YOLOv9 command: {command}")
    os.system(command)
    print(f"Results saved for YOLOv9 in {output_dir}/{model_name}")


def parse_detections(label_path):
    """Parse YOLO detection results."""
    boxes, scores, labels = [], [], []
    if not os.path.exists(label_path):
        return boxes, scores, labels

    with open(label_path, "r") as f:
        for line in f:
            try:
                class_id, x_center, y_center, width, height, confidence = map(float, line.strip().split())
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                boxes.append([x_min, y_min, x_max, y_max])  # Already normalized
                scores.append(confidence)
                labels.append(int(class_id))
            except ValueError:
                print(f"Warning: Malformed line in {label_path}")
    return boxes, scores, labels


def denormalize_boxes(boxes, width, height):
    """Convert normalized boxes to pixel coordinates."""
    return [
        [
            int(box[0] * width),  # x_min
            int(box[1] * height),  # y_min
            int(box[2] * width),  # x_max
            int(box[3] * height)  # y_max
        ]
        for box in boxes
    ]


def apply_wbf(boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.001):
    """Perform Weighted Box Fusion."""
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    return boxes, scores, labels

def draw_boxes_on_image(image, boxes, scores, labels):
    """Draw bounding boxes on the image with a white background behind the text."""
    for box, score, label in zip(boxes, scores, labels):
        x_min, y_min, x_max, y_max = box
        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Prepare text
        label_text = f"Label: {int(label)} Score: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw a filled white rectangle behind the text
        cv2.rectangle(
            image,
            (x_min, y_min - text_height - baseline - 5),
            (x_min + text_width, y_min),
            (255, 255, 255),
            -1
        )
        
        # Overlay the text on top of the white rectangle
        cv2.putText(
            image,
            label_text,
            (x_min, y_min - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Black text
            2
        )
    return image


def visualize_results(image_path, output_images, wbf_image_path):
    """Visualize the model outputs and the final WBF result."""
    fig, axes = plt.subplots(1, len(output_images) + 1, figsize=(15, 10))

    # Plot each model's result
    for idx, (model_name, image) in enumerate(output_images.items()):
        axes[idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[idx].axis("off")
        axes[idx].set_title(model_name)

    # Plot the WBF result
    wbf_image = cv2.imread(wbf_image_path)
    axes[-1].imshow(cv2.cvtColor(wbf_image, cv2.COLOR_BGR2RGB))
    axes[-1].axis("off")
    axes[-1].set_title("WBF Result")

    plt.tight_layout()
    plt.show()


def main():
    # Paths and settings
    image_source = "/home/jose/Documents/advanced_ML/images/images/image1.jpg"
    output_dir = "./runs/detect"
    img_size = 640
    conf = 0.25
    wbf_output_path = "./runs/detect/wbf_result.jpg"

    models = {
        "yolov7": {
            "weights": "yolo-aaa.pt",
            "run_function": run_yolov7
        },
        "yolov7x": {
            "weights": "yolo-x-aaa.pt",
            "run_function": run_yolov7
        },
        "yolov9c": {
            "weights": "yolov9c-run2-AAA/weights/best.pt",
            "run_function": run_yolov9
        },
        "yolov9-e": {
            "weights": "yolov9e-run1-AAA3/weights/best.pt",
            "run_function": run_yolov9
        }
    }

    image_width, image_height = get_image_dimensions(image_source)
    all_boxes_list = []
    all_scores_list = []
    all_labels_list = []
    output_images = {}

    for model_name, model_info in models.items():
        print(f"Processing model: {model_name}")
        weights = model_info["weights"]
        run_function = model_info["run_function"]

        run_function(weights, image_source, conf, img_size, output_dir, model_name)

        label_dir = os.path.join(output_dir, model_name, "labels")
        model_boxes, model_scores, model_labels = [], [], []
        if os.path.exists(label_dir):
            for label_file in os.listdir(label_dir):
                label_path = os.path.join(label_dir, label_file)
                boxes, scores, labels = parse_detections(label_path)
                model_boxes.extend(boxes)
                model_scores.extend(scores)
                model_labels.extend(labels)

        all_boxes_list.append(model_boxes)
        all_scores_list.append(model_scores)
        all_labels_list.append(model_labels)

        # Generate image with drawn boxes for this model
        model_image = cv2.imread(image_source)
        pixel_boxes = denormalize_boxes(model_boxes, image_width, image_height)
        model_image = draw_boxes_on_image(model_image, pixel_boxes, model_scores, model_labels)
        output_images[model_name] = model_image

    print("Applying Weighted Box Fusion (WBF)...")
    wbf_boxes, wbf_scores, wbf_labels = apply_wbf(all_boxes_list, all_scores_list, all_labels_list)

    # Save WBF result image
    wbf_pixel_boxes = denormalize_boxes(wbf_boxes, image_width, image_height)
    wbf_image = cv2.imread(image_source)
    wbf_image = draw_boxes_on_image(wbf_image, wbf_pixel_boxes, wbf_scores, wbf_labels)
    cv2.imwrite(wbf_output_path, wbf_image)

    # Visualize results
    visualize_results(image_source, output_images, wbf_output_path)


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)


if __name__ == "__main__":
    main()
