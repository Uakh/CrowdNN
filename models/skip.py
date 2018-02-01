# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.output_crop = 0
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,48,3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(48,64,3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(160, 80, 1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(80, 1, 1))
        self.mp = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        x = self.layer1(x)
        x1 = x
        x = self.mp(x)
        x = self.layer2(x)
        x2 = self.up(x)
        x = self.mp(x)
        x = self.layer3(x)
        x3 = self.up(x)
        x = self.mp(x)
        x = self.layer4(x)
        x4 = self.up(x)
        x = torch.cat((x1,x2,x3,x4), 1)
        x = self.layer5(x)
        return(x)

    def set_upsample(self, size):
        self.up = nn.UpsamplingBilinear2d(size=size)
