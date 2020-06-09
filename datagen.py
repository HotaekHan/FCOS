'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg width height xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
import random
import numpy as np
import json
import os
# from PIL import Image, ImageDraw, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import albumentations as A

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from encoder import DataEncoder

class jsonDataset(data.Dataset):
    def __init__(self, path, classes, transform, input_image_size, num_crops, fpn_level, is_norm_reg_target, radius,
                 do_aug=False, view_image=False, min_cols=1, min_rows=1):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) image shorter side size.
          max_size: (int) maximum image longer side size.
        '''
        self.path = path
        self.classes = classes
        self.transform = transform
        self.input_size = input_image_size
        self.num_crops = num_crops
        self.view_img = view_image
        self.do_aug = do_aug
        self.fpn_level = fpn_level
        self.is_norm_reg_target = is_norm_reg_target
        self.radius = radius

        self.fnames = []
        self.offsets = []
        self.boxes = []
        self.labels = []

        self.num_classes = len(self.classes)

        self.label_map = dict()
        self.class_idx_map = dict()
        # 0 is background class
        for idx in range(0, self.num_classes):
            self.label_map[self.classes[idx]] = idx+1   # 0 is background
            self.class_idx_map[idx+1] = self.classes[idx]

        self.data_encoder = DataEncoder(image_size=self.input_size,
                                        num_classes=self.num_classes + 1,
                                        fpn_level=self.fpn_level,
                                        is_norm_reg_target=self.is_norm_reg_target)

        fp_read = open(self.path, 'r')
        gt_dict = json.load(fp_read)

        all_boxes = list()
        all_labels = list()
        all_img_path = list()

        # read gt files
        for gt_key in gt_dict:
            gt_data = gt_dict[gt_key][0]

            box = list()
            label = list()

            num_boxes = len(gt_data['labels'])

            img = cv2.imread(gt_data['image_path'])
            img_rows = img.shape[0]
            img_cols = img.shape[1]

            for iter_box in range(0, num_boxes):
                xmin = gt_data['boxes'][iter_box][0]
                ymin = gt_data['boxes'][iter_box][1]
                xmax = gt_data['boxes'][iter_box][2]
                ymax = gt_data['boxes'][iter_box][3]
                rows = ymax - ymin
                cols = xmax - xmin

                if xmin < 0 or ymin < 0:
                    if xmin == -1:
                        xmin = 0
                    elif ymin == -1:
                        ymin = 0
                    else:
                        print('negative coordinate: [xmin: ' + str(xmin) + ', ymin: ' + str(ymin) + ']')
                        print(gt_data['image_path'])

                if xmax >= img_cols or ymax >= img_rows:
                    if xmax == img_cols:
                        xmax = img_cols - 1
                    elif ymax == img_rows:
                        ymax = img_rows -1
                    else:
                        print('over maximum size: [xmax: ' + str(xmax) + ', ymax: ' + str(ymax) + ']')
                        print(gt_data['image_path'])

                if cols < min_cols:
                    print('cols is lower than ' + str(min_cols) + ': [' + str(xmin) + ', ' + str(ymin) + ', ' +
                          str(xmax) + ', ' + str(ymax) + '] '
                          + str(gt_data['image_path']))
                    continue
                if rows < min_rows:
                    print('rows is lower than ' + str(min_rows) + ': [' + str(xmin) + ', ' + str(ymin) + ', ' +
                          str(xmax) + ', ' + str(ymax) + '] '
                          + str(gt_data['image_path']))
                    continue

                class_name = gt_data['labels'][iter_box][0]
                if class_name not in self.label_map:
                    print('weired class name: ' + class_name)
                    print(gt_data['image_path'])
                    continue

                class_idx = self.label_map[class_name]
                box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                label.append(int(class_idx))

            if len(box) == 0 or len(label) == 0:
                print('none of object exist in the image: ' + gt_data['image_path'])
                continue

            all_boxes.append(box)
            all_labels.append(label)
            all_img_path.append(gt_data['image_path'])

        if len(all_boxes) == len(all_labels) and len(all_boxes) == len(all_img_path):
            num_images = len(all_img_path)
        else:
            print('num. of boxes: ' + str(len(all_boxes)))
            print('num. of labels: ' + str(len(all_labels)))
            print('num. of paths: ' + str(len(all_img_path)))
            raise ValueError('num. of elements are different(all boxes, all_labels, all_img_path)')

        if num_crops <= 0:
            for idx in range(0, num_images, 1):
                self.fnames.append(all_img_path[idx])
                self.boxes.append(torch.tensor(all_boxes[idx], dtype=torch.float32))
                self.labels.append(torch.tensor(all_labels[idx], dtype=torch.int64))
        else:
            for idx in range(0, num_images, 1):
                ori_boxes = all_boxes[idx]
                ori_labels = all_labels[idx]

                ori_img = cv2.imread(all_img_path[idx])
                img_rows = ori_img.shape[0]
                img_cols = ori_img.shape[1]

                offsets, crop_boxes, crop_labels = self._do_crop(ori_img_rows=img_rows, ori_img_cols=img_cols,
                                                                 target_img_size=self.input_size,
                                                                 boxes=ori_boxes, labels=ori_labels)

                num_offsets = len(offsets)

                for idx_offset in range(0, num_offsets, 1):
                    self.fnames.append(all_img_path[idx])
                    self.offsets.append(offsets[idx_offset])
                    self.boxes.append(torch.tensor(crop_boxes[idx_offset], dtype=torch.float32))
                    self.labels.append(torch.tensor(crop_labels[idx_offset], dtype=torch.int64))

        self.num_samples = len(self.fnames)

        # if self.do_aug is True:
        bbox_params = A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.3)
        self.augmentation = A.Compose([
            # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            # A.GaussNoise(p=0.1),
            # A.GaussianBlur(blur_limit=3, p=0.05),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5),
            A.ChannelShuffle(p=1.0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.15, 0.15), rotate_limit=45, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(p=0.5),
        ], bbox_params=bbox_params, p=1.0)

    def __getitem__(self, idx):
        # Load image and boxes.
        fname = self.fnames[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        img = cv2.imread(fname)

        if self.num_crops <= 0:
            img, boxes = self._resize(img, boxes)
        else:
            offset = self.offsets[idx]
            crop_rect = (int(offset[0]), int(offset[1]), int(offset[0]+self.input_size[1]), int(offset[1]+self.input_size[0]))

            if offset[0] < 0 or offset[1] < 0:
                raise ValueError("negative offset!")

            for box in boxes:
                if box[0] < 0 or box[1] < 0 or box[2] > self.input_size[1] or box[3] > self.input_size[0]:
                    raise ValueError("negative box coordinate!")

            img = img[crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]

        if self.do_aug is True:
            bboxes = [bbox.tolist() + [label.item()] for bbox, label in zip(boxes, labels)]
            augmented = self.augmentation(image=img, bboxes=bboxes)
            img = np.ascontiguousarray(augmented['image'])
            boxes = augmented['bboxes']
            boxes = [list(bbox) for bbox in boxes]
            labels = [bbox.pop() for bbox in boxes]
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        if self.view_img is True:
            for idx_box, box in enumerate(boxes):
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0))
                class_idx = labels[idx_box].item()
                text_size = cv2.getTextSize(self.class_idx_map[class_idx], cv2.FONT_HERSHEY_PLAIN, 1, 1)
                cv2.putText(img, self.class_idx_map[class_idx], (box[0], box[1] - text_size[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            cv2.imwrite(os.path.join("crop_test", str(idx)+".jpg"), img)
            # mask_img = np.uint8(mask * 255.)
            # cv2.imwrite(os.path.join("crop_test", str(idx)+"_mask.jpg"), mask_img)

        img = self.transform(img)

        return img, boxes, labels, fname

    def __len__(self):
        return self.num_samples

    def _resize(self, img, boxes):
        if isinstance(self.input_size, int) is True:
            w = h = self.input_size
        elif isinstance(self.input_size, tuple) is True:
            h = self.input_size[0]
            w = self.input_size[1]
        else:
            raise ValueError('input size should be int or tuple of ints')

        ws = 1.0 * w / img.shape[1]
        hs = 1.0 * h / img.shape[0]
        scale = torch.tensor([ws, hs, ws, hs], dtype=torch.float32)
        if boxes.numel() == 0:
            scaled_box = boxes
        else:
            scaled_box = scale * boxes
        return cv2.resize(img, (w, h)), scaled_box

    def _do_crop(self, ori_img_rows, ori_img_cols, target_img_size, boxes, labels):
        num_boxes = len(boxes)
        num_labels = len(labels)

        if num_boxes != num_labels:
            print("error occur: Random crop")

        rand_indices = [0, 1, 2, 3, 4]
        np.random.shuffle(rand_indices)

        output_offsets = []
        output_boxes = []
        output_labels = []

        for box in boxes:
            # box coordinate from 1. not 0.
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]

            width = (xmax - xmin)+1
            height = (ymax - ymin)+1

            if width < 0 or height< 0:
                print("negative width/height")
                continue

            for iter_crop in range(0, self.num_crops, 1):
                rand_idx = rand_indices[iter_crop]

                margin = np.random.randint(16, 128, size=1)

                # top-left
                if rand_idx == 0:
                    offset_x = xmin-1-margin[0]
                    offset_y = ymin-1-margin[0]
                    crop_maxx = offset_x + target_img_size[1]
                    crop_maxy = offset_y + target_img_size[0]

                    if crop_maxx > ori_img_cols-1 or crop_maxy > ori_img_rows-1:
                        continue
                    if offset_x < 0 or offset_y < 0:
                        continue

                    crop_rect = [offset_x, offset_y, target_img_size[1], target_img_size[0]]

                    in_boxes, in_labels = self._find_boxes_in_crop(crop_rect, boxes, labels)

                    if len(in_boxes) == 0:
                        continue

                    output_offsets.append([offset_x, offset_y])
                    output_boxes.append(in_boxes)
                    output_labels.append(in_labels)
                # top-right
                elif rand_idx == 1:
                    offset_x = xmin - (target_img_size[1] - width)-1+margin[0]
                    offset_y = ymin-1-margin[0]
                    crop_maxx = offset_x + target_img_size[1]
                    crop_maxy = offset_y + target_img_size[0]

                    if crop_maxx > ori_img_cols-1 or crop_maxy > ori_img_rows-1:
                        continue

                    if offset_x < 0 or offset_y < 0:
                        continue

                    crop_rect = [offset_x, offset_y, target_img_size[1], target_img_size[0]]

                    in_boxes, in_labels = self._find_boxes_in_crop(crop_rect, boxes, labels)

                    if len(in_boxes) == 0:
                        continue

                    output_offsets.append([offset_x, offset_y])
                    output_boxes.append(in_boxes)
                    output_labels.append(in_labels)
                # bottom-left
                elif rand_idx == 2:
                    offset_x = xmin-1-margin[0]
                    offset_y = ymin - (target_img_size[0] - height)-1+margin[0]
                    crop_maxx = offset_x + target_img_size[1]
                    crop_maxy = offset_y + target_img_size[0]

                    if crop_maxx > ori_img_cols-1 or crop_maxy > ori_img_rows-1:
                        continue

                    if offset_x < 0 or offset_y < 0:
                        continue

                    crop_rect = [offset_x, offset_y, target_img_size[1], target_img_size[0]]

                    in_boxes, in_labels = self._find_boxes_in_crop(crop_rect, boxes, labels)

                    if len(in_boxes) == 0:
                        continue

                    output_offsets.append([offset_x, offset_y])
                    output_boxes.append(in_boxes)
                    output_labels.append(in_labels)
                # bottom-right
                elif rand_idx == 3:
                    offset_x = xmin - (target_img_size[1] - width)-1+margin[0]
                    offset_y = ymin - (target_img_size[0] - height)-1+margin[0]
                    crop_maxx = offset_x + target_img_size[1]
                    crop_maxy = offset_y + target_img_size[0]

                    if crop_maxx > ori_img_cols-1 or crop_maxy > ori_img_rows-1:
                        continue

                    if offset_x < 0 or offset_y < 0:
                        continue

                    crop_rect = [offset_x, offset_y, target_img_size[1], target_img_size[0]]

                    in_boxes, in_labels = self._find_boxes_in_crop(crop_rect, boxes, labels)

                    if len(in_boxes) == 0:
                        continue

                    output_offsets.append([offset_x, offset_y])
                    output_boxes.append(in_boxes)
                    output_labels.append(in_labels)
                # center
                elif rand_idx == 4:
                    rand_direction = np.random.randint(-1, 1, size=1)

                    offset_x = (xmin - ((target_img_size[1]-width)/2)-1) + (rand_direction[0] * margin[0])
                    offset_y = (ymin - ((target_img_size[0]-height)/2)-1) + (rand_direction[0] * margin[0])
                    crop_maxx = offset_x + target_img_size[1]
                    crop_maxy = offset_y + target_img_size[0]

                    if crop_maxx > ori_img_cols-1 or crop_maxy > ori_img_rows-1:
                        continue

                    if offset_x < 0 or offset_y < 0:
                        continue

                    crop_rect = [offset_x, offset_y, target_img_size[1], target_img_size[0]]

                    in_boxes, in_labels = self._find_boxes_in_crop(crop_rect, boxes, labels)

                    if len(in_boxes) == 0:
                        continue

                    output_offsets.append([offset_x, offset_y])
                    output_boxes.append(in_boxes)
                    output_labels.append(in_labels)

                else:
                    print("exceed possible crop num")

        return output_offsets, output_boxes, output_labels


    def _find_boxes_in_crop(self, crop_rect, boxes, labels):
        num_boxes = len(boxes)
        num_labels = len(labels)

        if num_boxes != num_labels:
            print("error occur: Random crop")

        boxes_in_crop=[]
        labels_in_crop = []
        for idx in range(0, num_boxes, 1):
            box_in_crop, label, is_contain = self._find_box_in_crop(crop_rect, boxes[idx], labels[idx])

            if is_contain is True:
                boxes_in_crop.append(box_in_crop)
                labels_in_crop.append(label)

        return boxes_in_crop, labels_in_crop


    def _find_box_in_crop(self, rect, box, label):
        rect_minx = rect[0]
        rect_miny = rect[1]
        rect_width = rect[2]
        rect_height = rect[3]

        box_minx = box[0]
        box_miny = box[1]
        box_maxx = box[2]
        box_maxy = box[3]
        box_width = (box_maxx - box_minx)+1
        box_height = (box_maxy - box_miny)+1

        # occlusion_ratio
        occlusion_ratio = 0.3
        occlusion_width = int(box_width * occlusion_ratio) * -1
        occlusion_height = int(box_height * occlusion_ratio) * -1

        box_in_crop_minx = box_minx - rect_minx
        if box_in_crop_minx <= occlusion_width or box_in_crop_minx >= rect_width:
            box_in_rect = []
            return box_in_rect, label, False

        box_in_crop_miny = box_miny - rect_miny
        if box_in_crop_miny <= occlusion_height or box_in_crop_miny >= rect_height:
            box_in_rect = []
            return box_in_rect, label, False

        box_in_crop_maxx = box_maxx - rect_minx
        if rect_width - box_in_crop_maxx <= occlusion_width or box_in_crop_maxx <= 0:
            box_in_rect = []
            return box_in_rect, label, False

        box_in_crop_maxy = box_maxy - rect_miny
        if rect_height - box_in_crop_maxy <= occlusion_height or box_in_crop_maxy <= 0:
            box_in_rect = []
            return box_in_rect, label, False

        if box_in_crop_minx < 0:
            box_in_crop_minx = 0
        if box_in_crop_miny < 0:
            box_in_crop_miny = 0
        if rect_width - box_in_crop_maxx < 0:
            box_in_crop_maxx = rect_width-1
        if rect_height - box_in_crop_maxy < 0:
            box_in_crop_maxy = rect_height-1

        box_in_rect = [box_in_crop_minx, box_in_crop_miny, box_in_crop_maxx, box_in_crop_maxy]
        return box_in_rect, label, True


    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        paths = [x[3] for x in batch]

        num_imgs = len(imgs)

        if isinstance(self.input_size, int) is True:
            inputs = torch.zeros([num_imgs, 3, self.input_size, self.input_size], dtype=torch.float32)
        elif isinstance(self.input_size, tuple) is True:
            inputs = torch.zeros([num_imgs, 3, self.input_size[0], self.input_size[1]], dtype=torch.float32)
        else:
            raise ValueError('input size should be int or tuple of ints')

        loc_targets = list()
        cls_targets = list()
        center_targets = list()

        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i, :, :imh, :imw] = im

            # Encode data.
            loc_target, cls_target, center_target = self.data_encoder.encode(boxes[i], labels[i], radius=self.radius)

            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
            center_targets.append(center_target)

        return inputs, \
               torch.stack(loc_targets, dim=0), \
               torch.stack(cls_targets, dim=0), \
               torch.stack(center_targets, dim=0), \
               paths


def test():
    import torchvision

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    # ])
    # set random seed
    random.seed(3000)
    np.random.seed(3000)
    torch.manual_seed(3000)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    classes = 'person|bicycle|car|motorcycle|bus|truck|cat|dog|rider'
    classes = classes.split('|')

    dataset = jsonDataset(path='data/voc.json', classes=classes,transform=transform,
                          input_image_size=(256, 512), num_crops=-1, fpn_level=5, is_norm_reg_target=True, radius=0.8,
                          view_image=True, do_aug=True)
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                                             collate_fn=dataset.collate_fn)

    for idx, (images, loc_targets, cls_targets, center_targets, paths) in enumerate(dataloader):
        print(loc_targets.shape)
        print(cls_targets.shape)
        print(center_targets.shape)
        pos_ind = cls_targets[:, :, 0] <= 0
        print(pos_ind.shape)
        print(pos_ind.data.long().sum())

if __name__ == '__main__':
    test()