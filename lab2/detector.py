import torch.nn as nn
import resnet


# TODO Design the detector.
# tips: Use pretrained `resnet` as backbone.

class Detector(nn.Module):

    def __init__(self, backbone: str='resnet50', lengths: tuple=(2048 * 4 * 4, 2048, 512),
                num_classes: int=5):
        super(Detector, self).__init__()

        resnet = eval('resnet.'+backbone)(pretrained = True, last_fc = True)
        in_features = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        for c in self.resnet.children():
            for para in c.parameters():
                para.requires_grad = False

        self.fc_classifier = nn.Linear(in_features, num_classes)
        self.fc_detector = nn.Linear(in_features, 4)

    def forward(self, X):
        out = self.resnet(X)
        out = out.squeeze()
        return self.fc_classifier(out), self.fc_detector(out)

# End of todo
