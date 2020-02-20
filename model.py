import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_filters, out_filters, stride=1, kernel_size=3, padding=1):
    return nn.Conv2d(in_filters, out_filters, stride=stride, kernel_size=kernel_size, padding=padding)

class Net(nn.Module):
    '''
    Small 3 Conv net with 1 FC layer
    Can be performed with or without Batchnorm and Dropout
    '''

    def __init__(self, pooling=nn.AvgPool2d, input_size=28, with_bn=False, with_dropout=False):
        super(Net, self).__init__()
        self.with_bn = with_bn
        self.with_dropout = with_dropout
        self.dropout = nn.Dropout2d(0.2)
        self.input_size = input_size
        self.conv1 = conv(1, 64)
        self.pool1 = pooling(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = conv(64, 128)
        self.pool2 = pooling(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv(128, 256)
        self.pool3 = pooling(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.linear = nn.Linear((input_size // 8) ** 2 * 256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        if self.with_bn:
            x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        if self.with_bn:
            x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        if self.with_bn:
            x = self.bn3(x)
        if self.with_dropout:
            x = self.dropout(x)

        x = x.view(-1, (self.input_size // 8) ** 2 * 256)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)