# -*- coding: utf-8 -*-
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.output_crop = 8 #How much output is cropped wrt input because of
                             #convolutions. Typically 0 if you use padding,
                             # >0 otherwise.
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,4, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,48,4, dilation=2),
            nn.BatchNorm2d(48),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 1, 1))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return(x)
