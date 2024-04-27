import numpy as np

def calculate_metrics(all_predictions, all_ground_truths):
    # Initialize variables
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    precision = 0
    recall = 0

    # Calculate precision and recall
    for pred_boxes, pred_labels in all_predictions:
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            found_match = False
            for gt_boxes, gt_labels in all_ground_truths:
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > 0.5 and pred_label == gt_label:
                        true_positives += 1
                        found_match = True
                        break
                if found_match:
                    break
            if not found_match:
                false_positives += 1

    for gt_boxes, gt_labels in all_ground_truths:
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            found_match = False
            for pred_boxes, pred_labels in all_predictions:
                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > 0.5 and pred_label == gt_label:
                        found_match = True
                        break
                if found_match:
                    break
            if not found_match:
                false_negatives += 1

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)

    # Calculate mAP (mean Average Precision)
    average_precision = 0
    num_ground_truths = len(all_ground_truths)

    for pred_boxes, pred_labels in all_predictions:
        precision_at_recall = []
        num_ground_truths_detected = 0

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            max_iou = 0
            for gt_boxes, gt_labels in all_ground_truths:
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > max_iou and pred_label == gt_label:
                        max_iou = iou
            if max_iou > 0.5:
                precision_at_recall.append(1)
                num_ground_truths_detected += 1
            else:
                precision_at_recall.append(0)

        precision_at_recall = [x for _,x in sorted(zip(pred_boxes, precision_at_recall), reverse=True)]
        precision_at_recall = np.cumsum(precision_at_recall) / (np.arange(len(precision_at_recall)) + 1)
        if num_ground_truths_detected > 0:
            average_precision += precision_at_recall[-1] / num_ground_truths_detected

    mAP = average_precision / num_ground_truths

    return precision, recall, mAP

def calculate_iou(box1, box2):
    # Calculate Intersection over Union (IoU) of two bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_x = max(x1, x2)
    intersection_y = max(y1, y2)
    intersection_w = max(0, min(x1 + w1, x2 + w2) - intersection_x)
    intersection_h = max(0, min(y1 + h1, y2 + h2) - intersection_y)

    intersection_area = intersection_w * intersection_h
    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0

    return iou