import math 

import torch
import torch.nn as nn


cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, num_classes, cfg, batch_norm):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg, batch_norm)
        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=True)]
                if batch_norm == True:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(inplace=True)]

                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
def vgg11(num_classes):
    return VGG(num_classes=num_classes, cfg=cfgs['VGG11'], batch_norm=False)

def vgg11_bn(num_classes):
    return VGG(num_classes=num_classes, cfg=cfgs['VGG11'], batch_norm=True)

def vgg13(num_classes):
    return VGG(num_classes=num_classes, cfg=cfgs['VGG13'], batch_norm=False)

def vgg13_bn(num_classes):
    return VGG(num_classes=num_classes, cfg=cfgs['VGG13'], batch_norm=True)

def vgg16(num_classes):
    return VGG(num_classes=num_classes, cfg=cfgs['VGG16'], batch_norm=False)

def vgg16_bn(num_classes):
    return VGG(num_classes=num_classes, cfg=cfgs['VGG16'], batch_norm=True)

def vgg19(num_classes):
    return VGG(num_classes=num_classes, cfg=cfgs['VGG19'], batch_norm=False)

def vgg19_bn(num_classes):
    return VGG(num_classes=num_classes, cfg=cfgs['VGG19'], batch_norm=True)
