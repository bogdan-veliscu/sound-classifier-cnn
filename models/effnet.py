import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class EffNet(nn.Module):
    def __init__(self, dataset, pretrained=True):
        super(EffNet, self).__init__()
        num_classes = 50 if dataset == "ESC" else 10
        self.model = EfficientNet.from_name("efficientnet-b1", num_classes)

    def forward(self, x):
        output = self.model(x)
        return output
