import torch
import torch.nn as nn
from torchvision import models


class BiomassModel(nn.Module):
    def __init__(self, n_outputs=1, model_name="resnet18"):
        super().__init__()

        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=None)

            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_outputs)
            )

        elif model_name == "mobilenetv2":
            self.backbone = models.mobilenet_v2(weights=None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_outputs)
            )

        elif model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_outputs)
            )

        else:
            raise ValueError("Invalid model name")

    def forward(self, x):
        return self.backbone(x)
