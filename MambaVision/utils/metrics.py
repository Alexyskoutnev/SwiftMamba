import numpy as np
from collections import namedtuple
from yolov5.models.common import Detections

pred_label = namedtuple('pred_label', ['label_class', 'xmin', 'ymin', 'xmax', 'ymax'])

def preprocess_yolo_labels(labels):
    objects = []
    for label in labels:
        for pred_tuple in label:
            pred_label_name = pred_tuple.label_class[0]
            xmin = pred_tuple.xmin.item()
            ymin = pred_tuple.ymin.item()
            xmax = pred_tuple.xmax.item()
            ymax = pred_tuple.ymax.item()
            obj = pred_label(pred_label_name, xmin, ymin, xmax, ymax)
            objects.append(obj)
    return objects

def preprocess_yolo_output(predictions):
    objects = []
    for predict in predictions:
        _objects = []
        pd_frame = predict.pandas().xyxy[0]
        for i in range(len(pd_frame)):
            xmin = pd_frame.iloc[i]['xmin']
            ymin = pd_frame.iloc[i]['ymin']
            xmax = pd_frame.iloc[i]['xmax']
            ymax = pd_frame.iloc[i]['ymax']
            class_id = pd_frame.iloc[i]['class']
            class_name = pd_frame.iloc[i]['name']
            # if class_name == "car":
            obj = pred_label(class_name, xmin, ymin, xmax, ymax)
            _objects.append(obj)
        objects.append(_objects)
    return objects

def single_label(all_predictions, all_ground):
    all_predictions_post = []
    all_ground_post = []
    for pred, pred_ground in zip(all_predictions, all_ground):
        if len(pred) == 1:
            all_predictions_post.append(pred[0])
            all_ground_post.append(pred_ground)
        elif len(pred) == 0:
            all_predictions_post.append([])
            all_ground_post.append(pred_ground)
    return all_predictions_post, all_ground_post
        
def calculate_metrics(all_predictions, all_ground_truths, iou_threshold=0.80):
    if "Detections" in str(type(all_predictions[0])):
        all_predictions_post = preprocess_yolo_output(all_predictions)
        all_ground_truths_post = preprocess_yolo_labels(all_ground_truths)
        # all_predictions_post, all_ground_truths_post = single_label(all_predictions_post, all_ground_truths_post)


    true_positives = 0
    false_positives = 0
    false_negatives = 0
    precision = 0
    recall = 0

    for ground_truths in all_ground_truths_post:
        found_match = False
        found_false_positive = False
        for preds in all_predictions_post:
            for pred in preds:
                pred_box = (pred.xmin, pred.ymin, pred.xmax - pred.xmin, pred.ymax - pred.ymin)
                gt_box = (ground_truths.xmin, ground_truths.ymin, ground_truths.xmax - ground_truths.xmin, ground_truths.ymax - ground_truths.ymin)
                iou = calculate_iou(pred_box, gt_box)
                if iou > iou_threshold and pred.label_class == ground_truths.label_class:
                    if not found_match and not found_false_positive:
                        true_positives += 1
                        found_match = True
                elif iou > iou_threshold and pred.label_class != ground_truths.label_class:
                    if not found_match and not found_false_positive:
                        false_positives += 1
                        found_false_positive = True
                        
        if not found_match and not found_false_positive:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0  
    average_precision = precision * recall # assume we have 1 class

    return precision, recall, average_precision

def calculate_iou(box1, box2):
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