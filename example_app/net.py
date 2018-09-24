# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,padding=1),
            #nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            #nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            #nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256,256, kernel_size=3,padding=1),
            #nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            )

        # self.fc1 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        # print(output.shape)
        output = output.view(output.size()[0], -1)
        # output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class HardSwish(nn.Module):

    def __init__(self, mean=0, std=1, min=0.1, max=0.9):
        super(HardSwish, self).__init__()

    def forward(self, x):
        return x*torch.clamp(torch.clamp(x*0.2+0.5, max=1),min=0)

class SiameseNetwork_customactivation(nn.Module):
    def __init__(self):
        super(SiameseNetwork_customactivation, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,padding=1),
            HardSwish(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            HardSwish(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            HardSwish(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            HardSwish(),
            nn.MaxPool2d(2, stride=2),
            )

        # self.fc1 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        # print(output.shape)
        output = output.view(output.size()[0], -1)
        # output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
