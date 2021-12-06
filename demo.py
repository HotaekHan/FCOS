# python
import os
import argparse
import cv2
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# pytorch
import torch

# user-defined
from models.fcos import load_model
import utils
from encoder import DataEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
parser.add_argument('--imgs', type=str, required=True, help='path of image files')
opt = parser.parse_args()

config = utils.get_config(opt.config)

cls_th = float(config['inference']['cls_threshold'])
nms_th = 0.5

output_dir = os.path.join(config['model']['exp_path'], 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# cuda
if torch.cuda.is_available() and not config['cuda']['using_cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

cuda_str = 'cuda:' + str(config['cuda']['gpu_id'])
device = torch.device(cuda_str if config['cuda']['using_cuda'] else "cpu")

bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.3)

is_resized = True
target_classes = utils.read_txt(config['params']['classes'])
if isinstance(config['inference']['image_size'], str) == True:
    img_size = config['inference']['image_size'].split('x')
    img_size = (int(img_size[0]), int(img_size[1])) # rows x cols
    print('Image size(normalization): ' + config['inference']['image_size'])

    valid_transforms = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1], p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ], bbox_params=bbox_params, p=1.0)
else:
    img_size = None
    is_resized = False
    print('Do not normalize to image size')

    valid_transforms = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ], bbox_params=bbox_params, p=1.0)

num_classes = len(target_classes)

net = load_model(num_classes=num_classes,
                 fpn_level=5,
                 basenet=config['params']['base'],
                 is_pretrained_base=False,
                 is_norm_reg_target=config['params']['norm_reg_target'],
                 centerness_with_loc=config['params']['centerness_on_reg'],
                 is_train=False)
net = net.to(device)
net.eval()

data_encoder = DataEncoder(image_size=img_size,
                           num_classes=num_classes + 1,
                           fpn_level=5,
                           is_norm_reg_target=config['params']['norm_reg_target'])

ckpt = torch.load(os.path.join(config['model']['exp_path'], 'best.pth'), map_location=device)
weights = utils._load_weights(ckpt['net'])
missing_keys = net.load_state_dict(weights, strict=False)
print(missing_keys)

class_idx_map = dict()
for idx in range(0, num_classes):
    class_idx_map[idx + 1] = target_classes[idx]

img_paths = list()
for (path, _, files) in os.walk(opt.imgs):
    for file in files:
        ext = os.path.splitext(file)[-1]

        if ext == '.jpg':
            img_paths.append(os.path.join(path, file))

img_dir_name = os.path.split(opt.imgs)[-1]
result_dir = os.path.join(config['model']['exp_path'], 'results', img_dir_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True)

with torch.set_grad_enabled(False):
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        ori_rows = img.shape[0]
        ori_cols = img.shape[1]
        if is_resized is False:
            img_size = (ori_rows, ori_cols)
        resized_img = cv2.resize(img, (img_size[1], img_size[0]))
        augmented = valid_transforms(image=img)
        x = augmented['image']
        x = x.unsqueeze(0)
        x = x.to(device)

        loc_preds, cls_preds, center_preds = net(x)

        boxes, scores, labels = data_encoder.decode(loc_preds=loc_preds.squeeze(0),
                                                    cls_preds=cls_preds.squeeze(0),
                                                    center_preds=center_preds.squeeze(0),
                                                    input_size=img_size,
                                                    cls_threshold=cls_th,
                                                    nms_threshold=nms_th)

        utils._write_results(result_dir, img_path, boxes, scores, labels, class_idx_map, img_size)



