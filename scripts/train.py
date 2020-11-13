# train.py
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

import os
import sys

try:
    from r2handrilnet.R2HandRilDataset import R2HandRilDataset
    from r2handrilnet.train_function import train_function
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath('r2handrilnet'))))
    from r2handrilnet.R2HandRilDataset import R2HandRilDataset
    from r2handrilnet.train_function import train_function

if __name__ == '__main__':
    train_ds = R2HandRilDataset('../Dataset/Train')
    test_ds = R2HandRilDataset('../Dataset/Test')

    x_train = train_ds.data()
    y_train = train_ds.targets()
    x_test = test_ds.data()
    y_test = test_ds.targets()

    train_function(x_train, y_train, x_test, y_test)
