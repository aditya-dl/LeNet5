import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, pool="max", activation="relu"):
        super(LeNet, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # Second convolution layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # Third convolution layer
        # According to the paper, we have conv3 because if the input size to the network changes the mapping in this
        # layer will change from 1x1 (with 32x32 image size the mapping is 1x1 with output of 5x5 mapped with input
        # kernel of 5x5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        # First dense/linear/fully-connected layer
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        # Second dense/linear/fully-connected layer
        self.fc2 = nn.Linear(in_features=84, out_features=10)

        # choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU()

        # choose pooling
        if pool == "avg":
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pool == "max":
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
