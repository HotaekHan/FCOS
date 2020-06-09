import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Resnet import ResFPN50, ResFPN101
from models.Densenet import DenseFPN62, DenseFPN102

import math


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale

class FCOS(nn.Module):
    def __init__(self, num_classes, basenet, fpn_level,
                 centerness_with_loc,
                 is_norm_reg_target,
                 is_train,
                 is_pretrained_base=False):
        super(FCOS, self).__init__()
        self.num_classes = num_classes
        self.num_head_layers = 4
        self.fpn_level = fpn_level
        self.centerness_with_loc = centerness_with_loc
        self.is_norm_reg_target = is_norm_reg_target
        self.is_train = is_train

        if basenet == 'Res50':
            if is_pretrained_base is True:
                self.base_networks = ResFPN50(is_pretrained=True, use_se=False)
            else:
                self.base_networks = ResFPN50(is_pretrained=False, use_se=False)
        elif basenet == 'Res101':
            if is_pretrained_base is True:
                self.base_networks = ResFPN101(is_pretrained=True, use_se=False)
            else:
                self.base_networks = ResFPN101(is_pretrained=False, use_se=False)
        elif basenet == 'Dense62':
            self.base_networks = DenseFPN62(use_se=False, efficient=False)
        elif basenet == 'Dense102':
            self.base_networks = DenseFPN102(use_se=False, efficient=False)
        else:
            raise ValueError('not supported base network.')


        self.loc_head = self._make_head()
        self.cls_head = self._make_head()

        self.loc_out = nn.Conv2d(self.base_networks.output_dims, 4, kernel_size=3, stride=1, padding=1)
        self.cls_out = nn.Conv2d(self.base_networks.output_dims, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.center_out = nn.Conv2d(self.base_networks.output_dims, 1, kernel_size=3, stride=1, padding=1)

        self.scales = nn.ModuleList([ScaleLayer(init_value=1.0) for _ in range(self.fpn_level)])

        downsample_cnt = 3
        self.scale_factors = [pow(2., i + downsample_cnt) for i in range(self.fpn_level)]

    def forward(self, x):
        p3, p4, p5, p6, p7 = self.base_networks(x)

        fms = [p3, p4, p5, p6, p7]

        loc_preds = list()
        cls_preds = list()
        center_preds = list()
        for feat_level, fm in enumerate(fms):
            loc_head = self.loc_head(fm)
            cls_head = self.cls_head(fm)

            cls_pred = self.cls_out(cls_head)
            loc_pred = self.loc_out(loc_head)

            if self.is_norm_reg_target is True:
                loc_pred = F.relu(self.scales[feat_level](loc_pred), inplace=True)
                if self.is_train is False:
                    loc_pred = loc_pred * self.scale_factors[feat_level]
            else:
                loc_pred = torch.exp(self.scales[feat_level](loc_pred))

            if self.centerness_with_loc is True:
                center_pred = self.center_out(loc_head)
            else:
                center_pred = self.center_out(cls_head)

            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, self.num_classes)
            center_pred = center_pred.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 1)

            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
            center_preds.append(center_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1), torch.cat(center_preds, 1)

    def _make_head(self):
        layers = []
        for _ in range(self.num_head_layers):
            layers.append(nn.Conv2d(self.base_networks.output_dims, self.base_networks.output_dims,
                                    kernel_size=3, stride=1, padding=1))
            layers.append(nn.GroupNorm(32, self.base_networks.output_dims))    #num. of group is 32
            layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)


def load_model(num_classes, basenet='Res50', fpn_level=5,
               is_norm_reg_target=False,
               centerness_with_loc=False,
               is_train=True,
               is_pretrained_base=False,
               do_freeze=False):

    fcos = FCOS(num_classes=num_classes, basenet=basenet, fpn_level=fpn_level,
                centerness_with_loc=centerness_with_loc,
                is_norm_reg_target=is_norm_reg_target,
                is_train=is_train,
                is_pretrained_base=is_pretrained_base)

    # head initialize
    for modules in [fcos.cls_head, fcos.loc_head,
                    fcos.cls_out, fcos.loc_out,
                    fcos.center_out]:
        for m in modules.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
                torch.nn.init.constant_(m.bias, 0)

    # biased initialize for classification head layer.
    pi = 0.01
    nn.init.constant_(fcos.cls_out.bias, -math.log((1 - pi) / pi))

    # freeze layer
    if do_freeze is True:
        for name, layer in fcos.base_networks.named_children():
            if name == 'conv1' or name == 'layer1':
                for param in layer.parameters():
                    param.requires_grad = False

    return fcos


def test():
    net = load_model(9, basenet='Res50', is_pretrained_base=False)

    # freeze layer
    for name, layer in net.base_networks.named_children():
        if name == 'conv1' or name == 'layer1':
            for param in layer.parameters():
                param.requires_grad = False

    num_parameters = 0.
    for param in net.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        num_parameters += num_layer_param

    base_num_parameters = 0.
    for param in net.base_networks.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        base_num_parameters += num_layer_param

    print(net)
    print("num. of parameters : " + str(num_parameters))
    print("num. of basenet parameters : " + str(base_num_parameters))

    loc_preds, cls_preds, center_preds = net(torch.randn(6, 3, 256, 512))


    print(loc_preds.size())
    print(cls_preds.size())
    print(center_preds.size())
    loc_grads = torch.randn(loc_preds.size())
    cls_grads = torch.randn(cls_preds.size())
    center_grads = torch.randn(center_preds.size())
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)
    center_preds.backward(center_grads)


if __name__ == '__main__':
    test()
