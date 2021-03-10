'''Some helper functions for PyTorch.'''

import torch
import yaml
import os
import collections
import cv2

def sort_with_indices(values, indices):
    num_elem = values.numel()

    '''bubble sort'''
    for current in range(0, num_elem, 1):
        for next_idx in range(current+1, num_elem, 1):
            if values[next_idx] > values[current]:
                tmp_value = values[current].item()
                tmp_idx = indices[current].item()
                values[current] = values[next_idx].item()
                indices[current] = indices[next_idx].item()
                values[next_idx] = tmp_value
                indices[next_idx] = tmp_idx


def box_nms(bboxes, scores, nms_threshold, mode, ovr_mode='union', soft_threshold=0.01):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      nms_threshold: (float) overlap threshold.
      cls_threshold: (float) classification threshold.
      mode: (str) 'soft' or 'hard'.
      ovr_mode: (str) 'union' or 'min'.

    Returns:
      keep: (tensor) selected indices.
    '''
    sigma = 0.5

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2-x1+1) * (y2-y1+1)
    ordered_score, ordered_idx = scores.sort(0, descending=True)

    keep = []
    while ordered_idx.numel() > 0:
        if ordered_idx.numel() == 1:
            max_idx = ordered_idx.item()
            keep.append(max_idx)
            break

        max_idx = ordered_idx[0]
        keep.append(max_idx)

        xx1 = x1[ordered_idx[1:]].clamp(min=x1[max_idx].item())
        yy1 = y1[ordered_idx[1:]].clamp(min=y1[max_idx].item())
        xx2 = x2[ordered_idx[1:]].clamp(max=x2[max_idx].item())
        yy2 = y2[ordered_idx[1:]].clamp(max=y2[max_idx].item())

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if ovr_mode == 'union':
            ovr = inter / (areas[max_idx] + areas[ordered_idx[1:]] - inter)
        elif ovr_mode == 'min':
            ovr = inter / areas[ordered_idx[1:]].clamp(max=areas[max_idx].item())
        else:
            raise TypeError('Unknown nms mode: %s.' % ovr_mode)

        weights = torch.zeros_like(ovr)
        if mode == 0:
            # soft-nms(linear)
            ovr_idx = ovr > nms_threshold
            non_ovr_idx = torch.logical_not(ovr_idx)

            weights[ovr_idx] = 1.0 - ovr[ovr_idx]
            weights[non_ovr_idx] = 1.0
        elif mode == 1:
            # soft-nms(gaussian)
            weights = torch.exp(-1. * (torch.pow(ovr, 2) / sigma))
        else:
            # hard-nms
            ovr_idx = ovr > nms_threshold
            non_ovr_idx = torch.logical_not(ovr_idx)

            weights[ovr_idx] = 0.0
            weights[non_ovr_idx] = 1.0

        ordered_idx = ordered_idx[1:]
        ordered_score = weights * ordered_score[1:]
        ids = (ordered_score > soft_threshold)
        if ids.numel() == 0:
            break
        ordered_idx = ordered_idx[ids]
        ordered_score = ordered_score[ids]

        if mode < 2:
            # sort for soft-nms
            sort_with_indices(ordered_score, ordered_idx)
    return torch.LongTensor(keep)


def get_config(conf):
    with open(conf, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


def print_config(conf):
    print(yaml.dump(conf, default_flow_style=False, default_style=''))


def get_best_model(dir_path):
    ckpt_file = dict()
    minimum_loss = float('inf')
    minimum_file = ''

    for (path, dirs, files) in os.walk(dir_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]

            if ext == '.pth':
                load_pth = torch.load(os.path.join(path, filename), map_location='cpu')
                valid_loss = load_pth['loss']

                ckpt_idx = filename
                ckpt_idx = int(ckpt_idx.split("-")[-1].split(".")[0])

                ckpt_file[ckpt_idx] = valid_loss

                if valid_loss < minimum_loss:
                    minimum_loss = valid_loss
                    minimum_file = filename

    for idx in ckpt_file:
        print("ckpt-" + str(idx) + " " + str(ckpt_file[idx]))

    if minimum_file == '':
        return None

    return os.path.join(dir_path, minimum_file)

def _load_weights(weights_dict):
    key, value = list(weights_dict.items())[0]

    trained_data_parallel = False
    if key[:7] == 'module.':
        trained_data_parallel = True

    if trained_data_parallel is True:
        new_weights = collections.OrderedDict()
        for old_key in weights_dict:
            new_key = old_key[7:]
            new_weights[new_key] = weights_dict[old_key]
    else:
        new_weights = weights_dict

    return new_weights

def _get_box_color(class_name):
    if class_name == 'person':
        box_color = (10, 180, 10)
    elif class_name == 'bicycle':
        box_color = (10, 180, 180)
    elif class_name == 'car':
        box_color = (180, 180, 10)
    elif class_name == 'motorcycle':
        box_color = (10, 10, 180)
    else:
        box_color = (180, 10, 10)

    return box_color


def _get_class_name(class_idx, class_idx_map):
    return class_idx_map[class_idx]


def _draw_rects(img, boxes, scores, labels, class_idx_map, ws, hs):
    for box_idx, box in enumerate(boxes):
        pt1 = (int(box[0] * ws), int(box[1] * hs))
        pt2 = (int(box[2] * ws), int(box[3] * hs))
        class_name = _get_class_name(labels[box_idx]+1, class_idx_map)
        score = float(scores[box_idx][0])
        out_text = class_name + ':' + format(score, ".2f")
        box_color = _get_box_color(class_name)
        cv2.rectangle(img, pt1, pt2, box_color, 1)
        t_size = cv2.getTextSize(out_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        pt2 = pt1[0] + (t_size[0] + 3), pt1[1] - (t_size[1] + 4)
        cv2.rectangle(img, pt1, pt2, box_color, -1)
        cv2.putText(img, out_text, (pt1[0], pt1[1] - (t_size[1] - 7)), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

def _write_txt(out_path, boxes, scores, labels, class_idx_map, ws, hs):
    f_out = open(out_path, 'w')

    for box_idx, box in enumerate(boxes):
        pt1 = (int(box[0] * ws), int(box[1] * hs))
        pt2 = (int(box[2] * ws), int(box[3] * hs))
        class_name = _get_class_name(labels[box_idx] + 1, class_idx_map)
        score = scores[box_idx][0]

        out_txt = str(class_name) + '\t' + \
                  str(pt1[0]) + '\t' + str(pt1[1]) + '\t' + str(pt2[0]) + '\t' + str(pt2[1]) + '\t' \
                  + str(score) + '\n'
        f_out.write(out_txt)

    f_out.close()


def _write_results(dir_path, img_path, boxes, scores, labels, class_idx_map, input_size):
    if not isinstance(boxes, list):
        boxes = boxes.tolist()
        scores = scores.tolist()
        labels = labels.tolist()

    image_name = os.path.basename(img_path)
    image_ext = os.path.splitext(img_path)[-1]
    image_name = image_name.replace(image_ext, '')

    img = cv2.imread(img_path)
    resized_rows = input_size[0]
    resized_cols = input_size[1]
    ori_rows = img.shape[0]
    ori_cols = img.shape[1]
    ws = ori_cols / resized_cols
    hs = ori_rows / resized_rows
    _draw_rects(img, boxes=boxes, scores=scores, labels=labels, class_idx_map=class_idx_map, ws=ws, hs=hs)

    img_out = os.path.join(dir_path, image_name + image_ext)
    cv2.imwrite(img_out, img)
    _write_txt(os.path.join(dir_path, image_name + '.txt'), boxes=boxes, scores=scores, labels=labels,
               class_idx_map=class_idx_map, ws=ws, hs=hs)

def read_txt(txt_path):
    f_read = open(txt_path, 'r')
    lines = f_read.readlines()

    out = list()
    for line in lines:
        out.append(line.rstrip())

    return out



