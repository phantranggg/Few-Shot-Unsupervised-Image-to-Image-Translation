import torch.nn as nn
from models.resBlk import ResBlocks

class ContentEncoder(nn.Module):
    def __init__(self, input_dim=3):
        super(ContentEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.activation_1 = nn.ReLU(inplace=True)
        self.in_1 = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1)
        self.activation_2 = nn.ReLU(inplace=True)
        self.in_2 = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.activation_3 = nn.ReLU(inplace=True)
        self.in_3 = nn.InstanceNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.activation_4 = nn.ReLU(inplace=True)
        self.in_4 = nn.InstanceNorm2d(512)
        self.resblks = ResBlocks(2, 512, norm='in', activation='relu', pad_type='zero')

    def forward(self, x):
        out = self.conv1(x)
        out = self.in_1(out)
        out = self.activation_1(out)
        out = self.conv2(out)
        out = self.in_2(out)
        out = self.activation_2(out)
        out = self.conv3(out)
        out = self.in_3(out)
        out = self.activation_3(out)
        out = self.conv4(out)
        out = self.in_4(out)
        out = self.activation_4(out)
        out = self.resblks(out)
        return out
