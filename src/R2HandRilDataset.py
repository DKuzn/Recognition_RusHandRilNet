# R2HandRilDataset.py
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

from __future__ import print_function, division
import random as rd
import pathlib
import torch
from skimage import io
from torch.utils.data import Dataset


class R2HandRilDataset(Dataset):
    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        self.paths = self.__list_dirs()
        self.labels_names = self.__list_labels_names()
        self.label_indexes = self.__label_indexes()
        self.image_labels = self.__labels()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        image = io.imread(self.paths[item])
        image_and_label = {'image': torch.tensor(image), 'label': self.image_labels[item]}
        return image_and_label

    def __list_dirs(self):
        paths = list(self.root.glob('*/*'))
        paths = [str(path) for path in paths]
        rd.shuffle(paths)
        return paths

    def __list_labels_names(self):
        labels = sorted(item.name for item in self.root.glob('*/') if item.is_dir())
        return labels

    def __label_indexes(self):
        label_to_index = dict((name, index) for index, name in enumerate(self.labels_names))
        return label_to_index

    def __labels(self):
        image_labels = [self.label_indexes[pathlib.Path(path).parent.name] for path in self.paths]
        return image_labels

    def __list_tensors(self):
        images = [io.imread(self.paths[item]) for item in range(len(self.paths))]
        tensors = torch.tensor(images)
        return tensors

    def data(self):
        return self.__list_tensors()

    def targets(self):
        return torch.tensor(self.image_labels)
