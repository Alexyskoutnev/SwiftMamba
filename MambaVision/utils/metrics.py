import numpy as np
import torch
from collections import namedtuple
from yolov5.models.common import Detections

pred_label = namedtuple('pred_label', ['label_class', 'xmin', 'ymin', 'xmax', 'ymax', 'orig_h', 'orig_w'])

def label_2_class_name(label):
    label_to_class = {
    0: "n/a",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "street sign",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye glasses",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    45: "plate",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    66: "mirror",
    67: "dining table",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
    91: "hair brush"
    }
    return label_to_class[label]


def preprocess_yolo_labels(labels):
    objects = []
    for label in labels:
        for pred_tuple in label:
            pred_label_name = pred_tuple.label_class[0]
            xmin = pred_tuple.xmin.item()
            ymin = pred_tuple.ymin.item()
            xmax = pred_tuple.xmax.item()
            ymax = pred_tuple.ymax.item()
            orig_h = pred_tuple.orig_h.item()
            orig_w = pred_tuple.orig_w.item()
            obj = pred_label(pred_label_name, xmin, ymin, xmax, ymax, orig_h, orig_w)
            objects.append(obj)
    return objects

def preprocess_yolo_output(predictions, grond_truths):
    objects = []
    for predict, gt in zip(predictions, grond_truths):
        _objects = []
        pd_frame = predict.pandas().xyxy[0]
        orig_h = gt.orig_h
        orig_w = gt.orig_w
        for i in range(len(pd_frame)):
            xmin = pd_frame.iloc[i]['xmin']
            ymin = pd_frame.iloc[i]['ymin']
            xmax = pd_frame.iloc[i]['xmax']
            ymax = pd_frame.iloc[i]['ymax']
            class_id = pd_frame.iloc[i]['class']
            class_name = pd_frame.iloc[i]['name']
            obj = pred_label(class_name, xmin, ymin, xmax, ymax, orig_h, orig_w)
            _objects.append(obj)
        objects.append(_objects)
    return objects

def preprocess_vit_output(predictions, ground_truths):
    objects = []
    for predict, gt in zip(predictions, ground_truths):
        _objects = []
        pred_labels = predict['pred_logits'].argmax(axis=1)
        # pred_labels = torch.max(predict['pred_logits'], dim=2).indices[0]
        pred_boxes = predict['pred_boxes']
        orig_h = gt.orig_h
        orig_w = gt.orig_w
        for label, box in zip(pred_labels, pred_boxes):
            class_name = label_2_class_name(label.item())
            if isinstance(box, torch.Tensor):
                xmin = box[0].item()
                ymin = box[1].item()
                xmax = box[2].item()
                ymax = box[3].item()
            else:
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
            obj = pred_label(class_name, xmin, ymin, xmax, ymax, orig_h, orig_w)
            _objects.append(obj)
        objects.append(_objects)
    return objects

def preprocess_mamba(predictions, ground_truths):
    objects = []
    for predicts, gt in zip(predictions, ground_truths):
        print(f"type : {type(gt)}")
        try: 
            if type(gt) == list:
                gt = gt[0]
            _objects = []
            orig_h = gt.orig_h
            orig_w = gt.orig_w
            for predict in predicts:
                xmin = predict.xmin * orig_w
                ymin = predict.ymin * orig_h
                xmax = predict.xmax * orig_w
                ymax = predict.ymax * orig_h
                obj = pred_label(predict.label_class.lower(), xmin, ymin, xmax, ymax, orig_h, orig_w)
                _objects.append(obj)
            objects.append(_objects)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error in preprocess_mamba {predicts}")
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
        
def calculate_metrics(all_predictions, all_ground_truths, iou_threshold=0.50):
    if "Detections" in str(type(all_predictions[0])):
        all_ground_truths_post = preprocess_yolo_labels(all_ground_truths)
        all_predictions_post = preprocess_yolo_output(all_predictions, all_ground_truths_post)
        # all_predictions_post, all_ground_truths_post = single_label(all_predictions_post, all_ground_truths_post)
    elif isinstance(all_predictions[0], dict):
        all_ground_truths_post = preprocess_yolo_labels(all_ground_truths)
        all_predictions_post = preprocess_vit_output(all_predictions, all_ground_truths_post)   
    else:
        all_ground_truths_post = preprocess_yolo_labels(all_ground_truths)
        all_predictions_post = preprocess_mamba(all_predictions, all_ground_truths_post)


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
                    true_positives += 1
                    found_match = True
                elif iou > iou_threshold and pred.label_class != ground_truths.label_class:
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