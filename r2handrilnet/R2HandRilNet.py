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
            in_channels=1, out_channels=21, kernel_size=3, padding=0)
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=21, out_channels=21, kernel_size=3, padding=0)
        self.act2 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = torch.nn.Conv2d(
            in_channels=21, out_channels=62, kernel_size=3, padding=0)
        self.act3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(
            in_channels=62, out_channels=62, kernel_size=3, padding=0)
        self.act4 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = torch.nn.Conv2d(
            in_channels=62, out_channels=186, kernel_size=3, padding=0)
        self.act5 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = torch.nn.Linear(186, 186)
        self.act7 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(186, 62)

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


class LesserR2HandRilNet(torch.nn.Module):
    def __init__(self):
        super(LesserR2HandRilNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=7, kernel_size=3, padding=0)
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=7, out_channels=7, kernel_size=3, padding=0)
        self.act2 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = torch.nn.Conv2d(
            in_channels=7, out_channels=31, kernel_size=3, padding=0)
        self.act3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(
            in_channels=31, out_channels=31, kernel_size=3, padding=0)
        self.act4 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = torch.nn.Conv2d(
            in_channels=31, out_channels=124, kernel_size=3, padding=0)
        self.act5 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = torch.nn.Linear(124, 124)
        self.act7 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(124, 62)

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


class LeakyR2HandRilNet(torch.nn.Module):
    def __init__(self):
        super(LeakyR2HandRilNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=21, kernel_size=3, padding=0)
        self.act1 = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=21, out_channels=21, kernel_size=3, padding=0)
        self.act2 = torch.nn.LeakyReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = torch.nn.Conv2d(
            in_channels=21, out_channels=62, kernel_size=3, padding=0)
        self.act3 = torch.nn.LeakyReLU()
        self.conv4 = torch.nn.Conv2d(
            in_channels=62, out_channels=62, kernel_size=3, padding=0)
        self.act4 = torch.nn.LeakyReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = torch.nn.Conv2d(
            in_channels=62, out_channels=186, kernel_size=3, padding=0)
        self.act5 = torch.nn.LeakyReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = torch.nn.Linear(186, 186)
        self.act7 = torch.nn.LeakyReLU()

        self.fc2 = torch.nn.Linear(186, 62)

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


class BnormR2HandRilNet(torch.nn.Module):
    def __init__(self):
        super(BnormR2HandRilNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=21, kernel_size=3, padding=0)
        self.act1 = torch.nn.LeakyReLU()
        self.bnorm1 = torch.nn.BatchNorm2d(21)
        self.conv2 = torch.nn.Conv2d(
            in_channels=21, out_channels=21, kernel_size=3, padding=0)
        self.act2 = torch.nn.LeakyReLU()
        self.bnorm2 = torch.nn.BatchNorm2d(21)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = torch.nn.Conv2d(
            in_channels=21, out_channels=62, kernel_size=3, padding=0)
        self.act3 = torch.nn.LeakyReLU()
        self.bnorm3 = torch.nn.BatchNorm2d(62)
        self.conv4 = torch.nn.Conv2d(
            in_channels=62, out_channels=62, kernel_size=3, padding=0)
        self.act4 = torch.nn.LeakyReLU()
        self.bnorm4 = torch.nn.BatchNorm2d(62)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = torch.nn.Conv2d(
            in_channels=62, out_channels=186, kernel_size=3, padding=0)
        self.act5 = torch.nn.LeakyReLU()
        self.bnorm5 = torch.nn.BatchNorm2d(186)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = torch.nn.Linear(186, 186)
        self.act7 = torch.nn.LeakyReLU()
        self.bnorm6 = torch.nn.BatchNorm1d(186)

        self.fc2 = torch.nn.Linear(186, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bnorm1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bnorm2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.bnorm3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.bnorm4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.bnorm5(x)
        x = self.pool3(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act7(x)
        x = self.bnorm6(x)

        x = self.fc2(x)

        return x
