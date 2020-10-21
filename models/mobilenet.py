import torch
import torch.nn as nn
import torchvision.models as models


class MobileNet(nn.Module):
    def __init__(self, dataset, pretrained=True):
        super(MobileNet, self).__init__()
        num_classes = 50 if dataset == "ESC" else 10
        self.model = models.mobilenet_v2(pretrained=pretrained)

        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )

    def forward(self, x):
        output = self.model(x)
        return output
