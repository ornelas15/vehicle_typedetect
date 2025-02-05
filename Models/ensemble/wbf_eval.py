import os
from ensemble_boxes import weighted_boxes_fusion, nms, soft_nms, non_maximum_weighted
import cv2
import numpy as np
from PIL import Image


def run_yolov7(weights, source, conf, img_size, output_dir, model_name):
    """Run YOLOv7 inference and save results."""
    absolute_output_dir = os.path.abspath(output_dir)
    command = (
        f"cd ../yolov7 && "
        f"python detect.py --weights {weights} --source {source} --conf-thres {conf} --img-size {img_size} "
        f"--project {absolute_output_dir} --name {model_name} --exist-ok --save-txt --save-conf && cd .."
    )
    print(f"Running YOLOv7 command: {command}")
    os.system(command)
    print(f"Results saved for YOLOv7 in {absolute_output_dir}/{model_name}")



def run_yolov9(weights, source, conf, img_size, output_dir, model_name):
    """Run YOLOv9 inference directly."""
    yolov9_dir = "../yolov9"
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


def parse_ground_truth(label_path):
    """Parse ground truth labels in YOLO format."""
    boxes, labels = [], []
    if not os.path.exists(label_path):
        return boxes, labels

    with open(label_path, "r") as f:
        for line in f:
            try:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                boxes.append([x_min, y_min, x_max, y_max])  # Already normalized
                labels.append(int(class_id))
            except ValueError:
                print(f"Warning: Malformed line in {label_path}")
    return boxes, labels


def compare_to_ground_truth(boxes, labels, gt_boxes, gt_labels):
    """Evaluate detection results using precision, recall, mAP@50, and mAP@95 metrics."""
    from shapely.geometry import box as shapely_box
    import numpy as np

    def compute_iou(box1, box2):
        """Compute IoU between two boxes."""
        b1 = shapely_box(*box1)
        b2 = shapely_box(*box2)
        intersection = b1.intersection(b2).area
        union = b1.union(b2).area
        return intersection / union

    def compute_ap(recall, precision):
        """Compute Average Precision (AP) using the precision-recall curve."""
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))

        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        indices = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
        return ap

    # IoU thresholds for mAP@95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for iou_threshold in iou_thresholds:
        matched_gt = set()
        matched_predictions = []

        for i, (box, label) in enumerate(zip(boxes, labels)):
            # Find the best matching ground truth box
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                # Skip if GT box already matched or labels don't match
                if j in matched_gt or label != gt_label:
                    continue
                iou = compute_iou(box, gt_box)
                # If IoU is above the threshold, consider it a match
                if iou >= iou_threshold:
                    # Mark GT box as matched and add prediction to the list
                    matched_gt.add(j)
                    matched_predictions.append((i, iou))
                    break

        tp = [0] * len(boxes)
        fp = [0] * len(boxes)

        # Sort predictions by IoU
        for idx, _ in matched_predictions:
            tp[idx] = 1

        for i in range(len(boxes)):
            if i not in [idx for idx, _ in matched_predictions]:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros(len(tp))
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recall, precision)
        aps.append(ap)
    
 
    map_50 = aps[0]
    map_95 = np.mean(aps)


    return  map_50, map_95
    


def normalize_boxes(boxes):
    """Ensure all box coordinates are within [0, 1]."""
    return np.clip(boxes, 0, 1)


def prepare_for_ensemble(boxes_list, scores_list, labels_list):
    """Ensure consistency of input arrays for ensemble methods."""
    for i in range(len(boxes_list)):
        if len(boxes_list[i]) == 0:
            # If no detections, ensure an empty array of shape [0, 4]
            boxes_list[i] = np.zeros((0, 4))
            scores_list[i] = np.zeros((0,))
            labels_list[i] = np.zeros((0,))
        else:
            # Ensure boxes are normalized and converted to numpy arrays
            boxes_list[i] = normalize_boxes(np.array(boxes_list[i]))
            scores_list[i] = np.array(scores_list[i])
            labels_list[i] = np.array(labels_list[i])

    return boxes_list, scores_list, labels_list

def main():
    # Paths and settings
    # test_folder = "./vehicle_dataset/test"
    test_folder="./vehicle_dataset/valid/"
    image_folder = os.path.join(test_folder, "images")
    label_folder = os.path.join(test_folder, "labels")
    output_dir = "./runs/detect"
    img_size = 640
    conf = 0.25

    models = {
        "yolov7": {
            "weights": "yolov7.pt",
            "run_function": run_yolov7
        },
        "yolov7x": {
            "weights": "yolov7x.pt",
            "run_function": run_yolov7
        },
        "yolov9c": {
            "weights": "yolov9c.pt",
            "run_function": run_yolov9
        },
        "yolov9-e": {
            "weights": "yolov9e.pt",
            "run_function": run_yolov9
        }
    }
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    all_results = {"Models": {}, "WBF": [], "NMS": [], "Soft-NMS": [], "NMW": []}

    # Process each model
    for model_name, model_info in models.items():
        print(f"Processing model: {model_name}")
        model_info["run_function"](model_info["weights"], image_folder, conf, img_size, output_dir, model_name)
        model_results = []

        for image_file in image_files:
            label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + ".txt")
            gt_boxes, gt_labels = parse_ground_truth(label_path)

            det_path = os.path.join(output_dir, model_name, "labels", os.path.splitext(image_file)[0] + ".txt")
            model_boxes, model_scores, model_labels = parse_detections(det_path)

            map_50, map_95 = compare_to_ground_truth(model_boxes, model_labels, gt_boxes, gt_labels)
            model_results.append((map_50, map_95))
            print(f"Image: {image_file} | Model: {model_name} | mAP@50: {map_50:.2f}, mAP@95: {map_95:.2f}")

        all_results["Models"][model_name] = model_results

    # Print overall results for models after all have been processed
    print("\n\nOverall Results by Model:")
    for model_name, results in all_results["Models"].items():
        avg_map_50 = np.mean([r[0] for r in results]) if results else 0
        avg_map_95 = np.mean([r[1] for r in results]) if results else 0
        print(f"Model: {model_name} | mAP@50: {avg_map_50:.2f}, mAP@95: {avg_map_95:.2f}")

    # Ensemble methods (WBF, NMS, Soft-NMS, NMW)
    for image_file in image_files:
        label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + ".txt")
        gt_boxes, gt_labels = parse_ground_truth(label_path)

        boxes_list, scores_list, labels_list = [], [], []
        for model_name in models.keys():
            det_path = os.path.join(output_dir, model_name, "labels", os.path.splitext(image_file)[0] + ".txt")
            boxes, scores, labels = parse_detections(det_path)
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        # Prepare data for ensemble methods
        boxes_list, scores_list, labels_list = prepare_for_ensemble(boxes_list, scores_list, labels_list)

        # WBF
        wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list)
        map_50, map_95 = compare_to_ground_truth(wbf_boxes, wbf_labels, gt_boxes, gt_labels)
        all_results["WBF"].append((map_50, map_95))

        # NMS
        if any(len(boxes) > 0 for boxes in boxes_list):
            nms_boxes, nms_scores, nms_labels = nms(boxes_list, scores_list, labels_list)
            map_50, map_95 = compare_to_ground_truth(nms_boxes, nms_labels, gt_boxes, gt_labels)
            all_results["NMS"].append((map_50, map_95))
        else:
            all_results["NMS"].append((0, 0))

        # Soft-NMS
        if any(len(boxes) > 0 for boxes in boxes_list):
            soft_boxes, soft_scores, soft_labels = soft_nms(boxes_list, scores_list, labels_list)
            map_50, map_95 = compare_to_ground_truth(soft_boxes, soft_labels, gt_boxes, gt_labels)
            all_results["Soft-NMS"].append((map_50, map_95))
        else:
            all_results["Soft-NMS"].append((0, 0))

        # NMW
        if any(len(boxes) > 0 for boxes in boxes_list):
            nmw_boxes, nmw_scores, nmw_labels = non_maximum_weighted(boxes_list, scores_list, labels_list)
            map_50, map_95 = compare_to_ground_truth(nmw_boxes, nmw_labels, gt_boxes, gt_labels)
            all_results["NMW"].append((map_50, map_95))
        else:
            all_results["NMW"].append((0, 0))

    # Print overall results for ensemble methods
    print("\n\nOverall Results for Ensemble Methods:")
    for method, results in all_results.items():
        if method == "Models":
            continue
        avg_map_50 = np.mean([r[0] for r in results])
        avg_map_95 = np.mean([r[1] for r in results])
        print(f"{method} | mAP@50: {avg_map_50:.2f}, mAP@95: {avg_map_95:.2f}")

if __name__ == "__main__":
    main()
