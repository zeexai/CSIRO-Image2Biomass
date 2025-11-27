#use the pretrained model but changed only the head so we get the pure  feature vector to a linear nn model to predict our output 5 target classes
class SwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG.swin_model_name, pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, len(CFG.target_cols))

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)

class EffNetB1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG.effnet_model_name, pretrained=True, num_classes=0, global_pool="avg")
        self.head = nn.Linear(self.backbone.num_features, len(CFG.target_cols))

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)
