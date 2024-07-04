import torch
from torch import nn
from torch.nn import functional as F

from my_dataset import combine


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class MyResnet18(nn.Module):
    def __init__(self):
        super(MyResnet18, self).__init__()
        self.b1 = nn.Sequential(nn.Conv1d(12, 64, kernel_size=7, stride=3, padding=3),
                           nn.BatchNorm1d(64), nn.ReLU(),
                           nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.classifer = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                       nn.Flatten(),
                                       nn.Linear(512, 512), nn.ReLU(),
                                       nn.Linear(512, 64), nn.ReLU(),
                                       nn.Linear(64, 9))
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.classifer(x)
        return x


if __name__ =="__main__":
    input = torch.randn(32, 12, 10000)
    my_model = MyResnet18()
    output = my_model(input)
    print(output.shape)




