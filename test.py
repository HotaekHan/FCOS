# python
import os
import argparse
import random
import numpy as np
import shutil
import sys

# pytorch
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from timer import Timer

# user-defined
from models.fcos import load_model
from datagen import jsonDataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()

config = utils.get_config(opt.config)

cls_th = float(config['inference']['cls_threshold'])
nms_th = 0.5
warmup_period = 10

output_dir = os.path.join(config['model']['exp_path'], 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# cuda
if torch.cuda.is_available() and not config['cuda']['using_cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

cuda_str = 'cuda:' + str(config['cuda']['gpu_id'])
device = torch.device(cuda_str if config['cuda']['using_cuda'] else "cpu")

target_classes = utils.read_txt(config['params']['classes'])
num_classes = len(target_classes)
img_size = config['inference']['image_size'].split('x')
img_size = (int(img_size[0]), int(img_size[1]))

bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.3)
valid_transforms = A.Compose([
    A.Resize(height=img_size[0], width=img_size[1], p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
], bbox_params=bbox_params, p=1.0)

net = load_model(num_classes=num_classes,
                 fpn_level=5,
                 basenet=config['params']['base'],
                 is_pretrained_base=False,
                 is_norm_reg_target=config['params']['norm_reg_target'],
                 centerness_with_loc=config['params']['centerness_on_reg'],
                 is_train=False)
net = net.to(device)
net.eval()

ckpt_path = os.path.join(config['model']['exp_path'], 'best.pth')
print(ckpt_path)
ckpt = torch.load(ckpt_path, map_location=device)
weights = utils._load_weights(ckpt['net'])
missing_keys = net.load_state_dict(weights, strict=False)
if len(missing_keys) > 0:
    print('missing keys from weights dict')
    print(missing_keys)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'num. of params: {n_params}')

dataset_dict = config['data']
for dataset_name in dataset_dict:
    data_path = dataset_dict[dataset_name]
    if data_path.split(' ')[-1] == 'notest' or data_path is None:
        continue
    result_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    dataset = jsonDataset(path=data_path, classes=target_classes,
                          transform=valid_transforms,
                          input_image_size=img_size,
                          num_crops=-1,
                          is_norm_reg_target=config['params']['norm_reg_target'],
                          fpn_level=5,
                          radius=float(config['params']['radius']))
    assert dataset
    num_data = len(dataset)
    batch_size = int(config['inference']['batch_size'])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=0,
        collate_fn=dataset.collate_fn,
        pin_memory=True)

    timer_infer = Timer()
    timer_post = Timer()

    with torch.set_grad_enabled(False):
        for batch_idx, (inputs, loc_targets, cls_targets, center_targets, paths) in enumerate(data_loader):
            if batch_idx == warmup_period:
                timer_infer.reset()
                timer_post.reset()
            sys.stdout.write('\r' + str(batch_idx * batch_size) + ' / ' + str(num_data))
            inputs = inputs.to(device)

            torch.cuda.synchronize()
            timer_infer.tic()
            loc_preds, cls_preds, center_preds = net(inputs)
            torch.cuda.synchronize()
            timer_infer.toc()

            num_batch = loc_preds.shape[0]

            for iter_batch in range(num_batch):
                torch.cuda.synchronize()
                timer_post.tic()
                boxes, scores, labels = dataset.data_encoder.decode(loc_preds=loc_preds[iter_batch],
                                                                    cls_preds=cls_preds[iter_batch],
                                                                    center_preds=center_preds[iter_batch],
                                                                    input_size=img_size,
                                                                    cls_threshold=cls_th,
                                                                    nms_threshold=nms_th)
                torch.cuda.synchronize()
                timer_post.toc()

                utils._write_results(result_dir, paths[iter_batch], boxes, scores, labels, dataset.class_idx_map, img_size)

    print(f'device: {device}')
    print(f'mean. elapsed time(inference): {timer_infer.average_time * 1000.:.4f}')
    print(f'mean. elapsed time(post): {timer_post.average_time * 1000.:.4f}')







