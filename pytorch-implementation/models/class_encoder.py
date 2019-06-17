import torch.nn as nn

class ClassEncoder(nn.Module):
    def __init__(self, input_dim=3):
        super(ClassEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.activation_1 = nn.ReLU(inplace=True)
        self.in_1 = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.activation_2 = nn.ReLU(inplace=True)
        self.in_2 = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.activation_3 = nn.ReLU(inplace=True)
        self.in_3 = nn.InstanceNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.activation_4 = nn.ReLU(inplace=True)
        self.in_4 = nn.InstanceNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.activation_5 = nn.ReLU(inplace=True)
        self.in_5 = nn.InstanceNorm2d(1024)
        self.avg_pooling = nn.AvgPool2d(kernel_size=3, stride=2)

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
        out = self.conv5(out)
        out = self.in_5(out)
        out = self.activation_5(out)
        out = self.avg_pooling(out)
        return out

