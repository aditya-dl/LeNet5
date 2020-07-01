import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Second convolution layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avg2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Third convolution layer
        # According to the paper, we have conv3 because if the input size to the network changes the mapping in this
        # layer will change from 1x1 (with 32x32 image size the mapping is 1x1 with output of 5x5 mapped with input
        # kernel of 5x5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # First dense/linear/fully-connected layer
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        # Second dense/linear/fully-connected layer
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = nn.Tanh()(x)
        x = self.avg1(x)
        x = self.conv2(x)
        x = nn.Tanh()(x)
        x = self.avg2(x)
        x = self.conv3(x)
        x = nn.Tanh()(x)
        x = x.view(-1, LeNet.num_flat_features(x))
        x = self.fc1(x)
        x = nn.Sigmoid()(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

