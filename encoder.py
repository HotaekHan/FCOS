'''Encode object boxes and labels.'''
import math
import torch

from utils import box_nms


class DataEncoder:
    def __init__(self, image_size, num_classes, fpn_level, is_norm_reg_target):
        self.image_size = image_size
        downsample_cnt = 3

        self.scale_factors = [pow(2., i + downsample_cnt) for i in range(fpn_level)]
        self.maximum_distance = [-1, 64, 128, 256, 512, float('inf')]
        self.is_norm_reg_target = is_norm_reg_target

        self.num_classes = num_classes  # include backgroud idx(=0)

    def encode(self, boxes, labels, radius=1.0):
        '''
        boxes(Tensor) : all boxes of image.
        labels(Tensor) : all labels of image.
        '''

        loc_targets = list()
        cls_targets = list()
        center_targets = list()

        for scale_idx, scale in enumerate(self.scale_factors):
            # resized_boxes = boxes / scale

            cls_target = torch.zeros(self.num_classes, int(self.image_size[0] / scale), int(self.image_size[1] / scale),
                                     dtype=torch.float32)
            # first all position will be background
            cls_target[0, :, :] = 1

            loc_target = torch.zeros(4, int(self.image_size[0] / scale), int(self.image_size[1] / scale),
                                     dtype=torch.float32)
            loc_target[:, :, :] = 1e+7
            center_target = torch.zeros(1, int(self.image_size[0] / scale), int(self.image_size[1] / scale),
                                        dtype=torch.float32)

            for box_idx, box in enumerate(boxes):
                xmin = int(box[0].item())
                ymin = int(box[1].item())
                xmax = int(box[2].item())
                ymax = int(box[3].item())

                rows = ymax - ymin + 1
                cols = xmax - xmin + 1
                max_val = rows if rows > cols else cols

                # filter out object by scale
                if max_val < self.maximum_distance[scale_idx] or max_val > self.maximum_distance[scale_idx+1]:
                    continue

                # cls_target
                label = labels[box_idx].item()

                # central sampling
                roi_cols = int(cols * radius)
                roi_rows = int(rows * radius)
                sampled_xmin = int(xmin + (cols - roi_cols) / 2)
                sampled_ymin = int(ymin + (rows - roi_rows) / 2)
                sampled_xmax = int(xmax - (cols - roi_cols) / 2)
                sampled_ymax = int(ymax - (rows - roi_rows) / 2)

                # do round for quantization error.
                resized_xmin = int(sampled_xmin / scale + 0.5)
                resized_ymin = int(sampled_ymin / scale + 0.5)
                resized_xmax = int(sampled_xmax / scale + 0.5)
                resized_ymax = int(sampled_ymax / scale + 0.5)

                cls_target[label, resized_ymin:resized_ymax, resized_xmin:resized_xmax] = 1
                cls_target[0, resized_ymin:resized_ymax, resized_xmin:resized_xmax] = 0

                # loc_target(regress to lt-br)
                for cur_y in range(ymin, ymax, 1):
                    for cur_x in range(xmin, xmax, 1):
                        is_change = False
                        resized_cur_y = int(cur_y / scale + 0.5)
                        resized_cur_x = int(cur_x / scale + 0.5)
                        next_resized_cur_y = int((cur_y + 1) / scale + 0.5)
                        next_resized_cur_x = int((cur_x + 1) / scale + 0.5)
                        if resized_cur_y != next_resized_cur_y:
                            is_change = True
                        if resized_cur_x != next_resized_cur_x:
                            is_change = True

                        if is_change is False:
                            continue

                        l_star = cur_x - xmin
                        t_star = cur_y - ymin
                        r_star = xmax - cur_x
                        b_star = ymax - cur_y

                        if resized_cur_y >= loc_target.shape[1]:
                            resized_cur_y = loc_target.shape[1]-1
                        if resized_cur_x >= loc_target.shape[2]:
                            resized_cur_x = loc_target.shape[2]-1

                        # centerness target
                        min_lr = l_star if l_star < r_star else r_star
                        max_lr = l_star if l_star > r_star else r_star
                        min_tb = t_star if t_star < b_star else b_star
                        max_tb = t_star if t_star > b_star else b_star

                        if max_lr == 0 or max_tb == 0:
                            centerness = 0.0
                        else:
                            centerness = (min_lr / max_lr) * (min_tb / max_tb)
                        centerness = math.sqrt(centerness)

                        center_target[0, resized_cur_y, resized_cur_x] = centerness

                        if sampled_xmin <= cur_x <= sampled_xmax and sampled_ymin <= cur_y <= sampled_ymax:
                            # regression target
                            if self.is_norm_reg_target is True:
                                loc_target[0, resized_cur_y, resized_cur_x] = l_star / scale
                                loc_target[1, resized_cur_y, resized_cur_x] = t_star / scale
                                loc_target[2, resized_cur_y, resized_cur_x] = r_star / scale
                                loc_target[3, resized_cur_y, resized_cur_x] = b_star / scale
                            else:
                                loc_target[0, resized_cur_y, resized_cur_x] = l_star
                                loc_target[1, resized_cur_y, resized_cur_x] = t_star
                                loc_target[2, resized_cur_y, resized_cur_x] = r_star
                                loc_target[3, resized_cur_y, resized_cur_x] = b_star

            cls_target = cls_target.permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            loc_target = loc_target.permute(1, 2, 0).contiguous().view(-1, 4)
            center_target = center_target.permute(1, 2, 0).contiguous().view(-1, 1)

            cls_targets.append(cls_target)
            loc_targets.append(loc_target)
            center_targets.append(center_target)

        return torch.cat(loc_targets, 0), torch.cat(cls_targets, 0), torch.cat(center_targets, 0)

    def decode(self, loc_preds, cls_preds, center_preds, input_size, cls_threshold, nms_threshold):
        '''Decode outputs back to bouding box locations and class labels.
        Args:
          loc_preds: (tensor) predicted locations, sized [#batch, #samples, 4].
          cls_preds: (tensor) predicted class labels, sized [#batch, #samples , #classes].
          center_preds: (tensor) predicted centerness, sized [#batch, #samples, 1].
          input_size: (int/tuple) model input size of (h, w).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        scores, labels = cls_preds.sigmoid().max(1)
        pos_ind = scores > cls_threshold
        if pos_ind.to(dtype=torch.int8).nonzero().numel() == 0:
            return [], [], []

        center_preds = center_preds.sigmoid()
        scores = scores[:, None] * center_preds
        # scores = scores[:, None]

        # locations = (#batch, #samples, 2(x-y coordinate))
        locations = self._get_pixel_locations(input_size, loc_preds.device)

        boxes = loc_preds[pos_ind]
        locations = locations[pos_ind]
        scores = scores[pos_ind]
        labels = labels[pos_ind]

        boxes[:, 0] = locations[:, 0] - boxes[:, 0]
        boxes[:, 1] = locations[:, 1] - boxes[:, 1]
        boxes[:, 2] = locations[:, 0] + boxes[:, 2]
        boxes[:, 3] = locations[:, 1] + boxes[:, 3]

        # nms mode = 0: soft-nms(liner), 1: soft-nms(gaussian), 2: hard-nms
        keep = box_nms(boxes, scores, nms_threshold=nms_threshold, mode=2)

        return boxes[keep], scores[keep], labels[keep]

    def _get_pixel_locations(self, input_size, device):
        input_size = torch.tensor([input_size, input_size], dtype=torch.float32) if isinstance(input_size, int) \
            else torch.tensor(input_size, dtype=torch.float32)

        pixel_locations = list()
        for scale in self.scale_factors:
            feat_size = input_size / scale
            y = torch.arange(0, (feat_size[0] * scale).item(), step=scale, dtype=torch.float32, device=device)
            x = torch.arange(0, (feat_size[1] * scale).item(), step=scale, dtype=torch.float32, device=device)

            y, x = torch.meshgrid([y, x])
            x = x.reshape(-1)
            y = y.reshape(-1)
            pixel_location = torch.stack((x, y), dim=1) + scale // 2
            pixel_locations.append(pixel_location)

        return torch.cat(pixel_locations, 0).repeat(1, 1)





