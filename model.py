import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from dorefa import QConv2d, QReLU
import torch.nn.init as init
import torch.nn.functional as F
from model_serialization import load_state_dict


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


def conv3x3(in_planes, out_planes, bitW, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, bitW, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bitW, bitA, stride=1, downsample_booster=None, downsample_residual=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.bitW = bitW
        self.bitA = bitA
        self.conv1 = QConv2d(inplanes, planes, bitW, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, bitW, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QConv2d(planes, planes * 4, bitW, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.QReLU = QReLU(k=bitA) 
        self.downsample_booster = downsample_booster
        self.downsample = downsample_residual
        self.stride = stride
        self.auxiliary = nn.Sequential(conv1x1(planes * self.expansion, planes * self.expansion), nn.BatchNorm2d(planes * self.expansion))
        self.is_last = is_last


    def forward(self, x, booster):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.QReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.QReLU(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            booster = self.downsample_booster(booster)
            residual = self.downsample(x)

        out = out + residual
        booster = F.relu(booster + self.auxiliary(out))

        if not self.is_last:
            out = self.QReLU(out)
        else:
            out = F.relu(out)

        return out, booster



class ResNet(nn.Module):

    def __init__(self, block, layers, bitW, bitA, num_classes=1000):
        self.inplanes = 64
        self.bitW = bitW
        self.bitA = bitA
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.QReLU = QReLU(k=self.bitA)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_last=True)  #don't quantize the last layer
        self.avgpool = nn.AvgPool2d(7)
        self.fc_1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc_2 = nn.Linear(512 * block.expansion, num_classes)
        self.auxiliary_input = nn.Sequential(conv1x1(64, 64), nn.BatchNorm2d(64))


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, QConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample_booster = None
        downsample_residual = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_booster = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            downsample_residual = nn.Sequential(
                QConv2d(self.inplanes, planes * block.expansion, self.bitW,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = nn.ModuleList([])
        layers.append(block(self.inplanes, planes, self.bitW, self.bitA, stride, downsample_booster, downsample_residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, self.bitW, self.bitA))

        layers.append(block(self.inplanes, planes, self.bitW, self.bitA, is_last=is_last))


        return layers



    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        booster = self.auxiliary_input(x)

        x = self.QReLU(x)

        for layer in self.layer1:
            x, booster = layer(x, booster)

        for layer in self.layer2:
            x, booster = layer(x, booster)

        for layer in self.layer3:
            x, booster = layer(x, booster)

        for layer in self.layer4:
            x, booster = layer(x, booster)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)

        booster = self.avgpool(booster)
        booster = booster.view(booster.size(0), -1)
        booster = self.fc_2(booster)

        return x, booster


def resnet18(bitW, bitA, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], args.bitW, args.bitA, **kwargs)
    if args.resume_train:
        return model
    elif args.pretrained == True:
        load_dict = model_zoo.load_url(model_urls['resnet18'])
    else:
        load_dict = torch.load('./full_precision_weights/model_best.pth.tar')['state_dict'] 
    load_state_dict(model, load_dict) 
    return model


def resnet34(args, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], args.bitW, args.bitA, **kwargs)
    if args.resume_train:
        return model
    elif args.pretrained == True:
        load_dict = model_zoo.load_url(model_urls['resnet34'])
    else:
        load_dict = torch.load('./full_precision_weights/model_best.pth.tar')['state_dict'] 
    load_state_dict(model, load_dict) 
    return model


def resnet50(args, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], args.bitW, args.bitA, **kwargs)
    if args.resume_train:
        return model
    elif pretrained == True:
        load_dict = model_zoo.load_url(model_urls['resnet50'])
    else:
        load_dict = torch.load('./full_precision_weights/model_best.pth.tar')['state_dict'] 
    load_state_dict(model, load_dict) 
    return model
