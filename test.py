# python
import os
import argparse
import random
import numpy as np
import shutil
import sys

# pytorch
import torch
import torchvision.transforms as transforms

# user-defined
from models.fcos import load_model
from datagen import jsonDataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()

config = utils.get_config(opt.config)

cls_th = float(config['hyperparameters']['cls_threshold'])
nms_th = 0.5

output_dir = os.path.join(config['model']['exp_path'], 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# cuda
if torch.cuda.is_available() and not config['cuda']['using_cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

cuda_str = 'cuda:' + str(config['cuda']['gpu_id'])
device = torch.device(cuda_str if config['cuda']['using_cuda'] else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

target_classes = config['hyperparameters']['classes'].split('|')
img_size = config['hyperparameters']['image_size'].split('x')
img_size = (int(img_size[0]), int(img_size[1]))

num_classes = len(target_classes)

net = load_model(num_classes=num_classes,
                 fpn_level=5,
                 basenet=config['hyperparameters']['base'],
                 is_pretrained_base=False,
                 is_norm_reg_target=config['hyperparameters']['norm_reg_target'],
                 centerness_with_loc=config['hyperparameters']['centerness_on_reg'],
                 is_train=False)
net = net.to(device)
net.eval()

ckpt = torch.load(os.path.join(config['model']['exp_path'], 'latest.pth'), map_location=device)
weights = utils._load_weights(ckpt['net'])
missing_keys = net.load_state_dict(weights, strict=False)
if len(missing_keys) > 0:
    print('missing keys from weights dict')
    print(missing_keys)

dataset_dict = config['data']
for dataset_name in dataset_dict:
    data_path = dataset_dict[dataset_name]
    if data_path.split(' ')[-1] == 'notest':
        continue
    result_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    dataset = jsonDataset(path=data_path, classes=target_classes,
                          transform=transform,
                          input_image_size=img_size,
                          num_crops=-1,
                          is_norm_reg_target=config['hyperparameters']['norm_reg_target'],
                          fpn_level=5,
                          radius=float(config['hyperparameters']['radius']))
    assert dataset
    num_data = len(dataset)
    batch_size = int(config['hyperparameters']['batch_size'])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=0,
        collate_fn=dataset.collate_fn)

    with torch.set_grad_enabled(False):
        for batch_idx, (inputs, loc_targets, cls_targets, center_targets, paths) in enumerate(data_loader):
            sys.stdout.write('\r' + str(batch_idx * batch_size) + ' / ' + str(num_data))
            inputs = inputs.to(device)
            loc_preds, cls_preds, center_preds = net(inputs)
            num_batch = loc_preds.shape[0]

            for iter_batch in range(num_batch):
                boxes, scores, labels = dataset.data_encoder.decode(loc_preds=loc_preds[iter_batch],
                                                                    cls_preds=cls_preds[iter_batch],
                                                                    center_preds=center_preds[iter_batch],
                                                                    input_size=img_size,
                                                                    cls_threshold=cls_th,
                                                                    nms_threshold=nms_th)

                utils._write_results(result_dir, paths[iter_batch], boxes, scores, labels, dataset.class_idx_map, img_size)









