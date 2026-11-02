import torch.nn as nn
import math as math


class ResClassifier(nn.Module):
    def __init__(self, class_num, extract=False, training=False, dropout_p=0.5):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(152, 76),
            nn.BatchNorm1d(76, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )

        self.fc11 = nn.Linear(76, class_num)
        self.extract = extract
        self.training = training
        self.dropout_p = dropout_p

    def forward(self, x):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))
        logit = self.fc11(fc1_emb)
        if self.training:
            return logit, fc1_emb

        if self.extract:
            return fc1_emb, logit
        return logit


class Discriminator(nn.Module):
    def __init__(self, num_classes=12):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 100),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(100, affine=True),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        # x = self.classifier1(x)
        # print x
        x = self.classifier(x)
        return x
