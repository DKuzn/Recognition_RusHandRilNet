# train_function.py
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
import random
import json
import numpy as np
import os
import time
from .R2HandRilNet import R2HandRilNet, LesserR2HandRilNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_function(x_train, y_train, x_test, y_test, epochs=100, batch_size=100, net='main'):
    if net == 'main':
        model = R2HandRilNet()
    else:
        model = LesserR2HandRilNet()

    checkpoint_best = '../r2handrilnet/weights/best_weights.pt'
    checkpoint_last = '../r2handrilnet/weights/last_weights.pt'
    train_plot_path = '../log/train_plot.json'

    model = model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    x_train = x_train.unsqueeze(1).float()
    x_test = x_test.unsqueeze(1).float()

    train_accuracy_history = []
    train_loss_history = []
    test_accuracy_history = []
    test_loss_history = []

    less_loss = 1e+10
    start_time = time.time()

    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0

        order = np.random.permutation(len(x_train))
        for start_index in range(0, len(x_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index + batch_size]

            x_batch = x_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = model.forward(x_batch)
            train_accuracy = (preds.argmax(dim=1) == y_batch).float().mean().data.cpu()

            loss_value = loss_function(preds, y_batch)
            train_loss = loss_value

            loss_value.backward()

            optimizer.step()

        train_accuracy_history.append(float(train_accuracy))
        train_loss_history.append(float(train_loss))

        test_loss = 0
        test_accuracy = 0

        for start_index in range(0, len(x_test), batch_size):

            x_test_batch = x_test[start_index:start_index + batch_size].to(device)
            y_test_batch = y_test[start_index:start_index + batch_size].to(device)

            test_preds = model.forward(x_test_batch)

            test_accuracy = (test_preds.argmax(dim=1) == y_test_batch).float().mean().data.cpu()
            test_loss = loss_function(test_preds, y_test_batch).data.cpu()

        if test_loss < less_loss:
            torch.save(model.state_dict(), checkpoint_best)
            print('Test loss improve from {0} to {1}. Saving to {2}'.format(less_loss, test_loss, checkpoint_best))
            less_loss = test_loss
        else:
            print('Test loss did not improve from {0}.'.format(less_loss))

        test_loss_history.append(float(test_loss))
        test_accuracy_history.append(float(test_accuracy))

        print('Epoch {0}/{1} - train_loss: {2} - train_accuracy: {3} ' \
              '- test_loss: {4} - test_accuracy: {5}'.format(epoch + 1, epochs,
                                                             round(float(train_loss), 4),
                                                             round(float(train_accuracy), 4),
                                                             round(float(test_loss), 4),
                                                             round(float(test_accuracy), 4)))

    torch.save(model.state_dict(), checkpoint_last)
    end_time = time.time()
    learn_time = (end_time - start_time) / 3600
    print('Time to learn: ', learn_time, 'hours')

    if os.path.exists(train_plot_path):
        out_result = json.load(open(train_plot_path, 'r'))
    else:
        out_result = {'train_accuracy': [],
                      'test_accuracy': [],
                      'train_loss': [],
                      'test_loss': [],
                      'training_time': 0}

    out_result['train_accuracy'] += train_accuracy_history
    out_result['test_accuracy'] += test_accuracy_history
    out_result['train_loss'] += train_loss_history
    out_result['test_loss'] += test_loss_history
    out_result['training_time'] += learn_time
    with open(train_plot_path, 'w') as f:
        json.dump(out_result, f)
