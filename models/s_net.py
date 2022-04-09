import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf=64, nb=8):
        super(SaliencyNet, self).__init__()

        self.refpad01_1 = nn.ReflectionPad2d(in_nc)
        self.conv01_1 = nn.Conv2d(in_nc, nf, 7)
        self.bn01_1 = nn.InstanceNorm2d(64)
        # relu
        self.conv02_1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv02_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn02_1 = nn.InstanceNorm2d(128)
        # relu
        self.conv03_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv03_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn03_1 = nn.InstanceNorm2d(256)
        # relu

        self.deconv01_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv01_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn12_1 = nn.InstanceNorm2d(128)
        # relu
        self.deconv02_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv02_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn13_1 = nn.InstanceNorm2d(64)
        # relu
        self.refpad12_1 = nn.ReflectionPad2d(3)
        self.deconv03_1 = nn.Conv2d(64, out_nc, 7)
        # tanh

    def forward(self, x):

        output = []
        layer1 = F.relu(self.bn01_1(self.conv01_1(self.refpad01_1(x))))
        output.append(layer1)
        layer2 = F.relu(self.bn02_1(self.conv02_2(self.conv02_1(layer1))))
        output.append(layer2)
        layer3 = F.relu(self.bn03_1(self.conv03_2(self.conv03_1(layer2))))
        output.append(layer3)

        layer4 = F.relu(self.bn12_1(self.deconv01_2(self.deconv01_1(layer3))))
        layer5 = F.relu(self.bn13_1(self.deconv02_2(self.deconv02_1(layer4))))
        y = torch.tanh(self.deconv03_1(self.refpad12_1(layer5)))
        output.append(y)

        return output


