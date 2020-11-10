# R2HandRilNet.py
#
# Copyright 2020 Дмитрий Кузнецов
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch


class R2HandRilNet(torch.nn.Module):
    def __init__(self):
        super(R2HandRilNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=124, kernel_size=3, padding=0)
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=124, out_channels=124, kernel_size=3, padding=0)
        self.act2 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = torch.nn.Conv2d(
            in_channels=124, out_channels=596, kernel_size=3, padding=0)
        self.act3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(
            in_channels=596, out_channels=596, kernel_size=3, padding=0)
        self.act4 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = torch.nn.Conv2d(
            in_channels=596, out_channels=1192, kernel_size=3, padding=0)
        self.act5 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = torch.nn.Linear(1192, 1192)
        self.act7 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(1192, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.pool3(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act7(x)

        x = self.fc2(x)

        return x
