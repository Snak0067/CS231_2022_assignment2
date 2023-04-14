# -*- coding:utf-8 -*-
# @FileName :open_challenge.py
# @Time :2023/4/13 15:55
# @Author :Xiaofeng
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import torch.optim as optim
from PyTorch import flatten

from PyTorch import train_part34


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # TODO: Set up the layers you need for a three-layer ConvNet with the  #
        # architecture defined above.                                          #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.relu = F.relu

        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=channel_1, kernel_size=5, padding=2, bias=True)
        nn.init.kaiming_normal_(self.conv_1.weight)
        nn.init.constant_(self.conv_1.bias, 0)

        # self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_2 = nn.Conv2d(in_channels=channel_1, out_channels=channel_2, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv_2.weight)
        nn.init.constant_(self.conv_2.bias, 0)

        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)

    def forward(self, x):
        conv_1 = self.relu(self.conv_1(x))
        conv_2 = self.relu(self.conv_2(conv_1))
        scores = self.fc(flatten(conv_2))

        return scores


def Train_Three_Layer_Network():
    learning_rate = 1e-2
    channel_1 = 96
    channel_2 = 32

    model = ThreeLayerConvNet(3, channel_1, channel_2, 10)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    best_acc = train_part34(model, optimizer, epochs=10)
    print('best_accuracy: %.2f' % best_acc)


if __name__ == '__main__':
    Train_Three_Layer_Network()
