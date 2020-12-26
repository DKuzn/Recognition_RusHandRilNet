# prediction.py
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
import numpy as np
from .R2HandRilNet import R2HandRilNet, LesserR2HandRilNet, LeakyR2HandRilNet, BnormR2HandRilNet
from PIL import Image
import PIL.ImageOps


def predict_letter(image, invert=True, net='main'):
    if net == 'main':
        model = R2HandRilNet()
        model.load_state_dict(torch.load('r2handrilnet/weights/R2HandRilNet.pt'))
    elif net == 'leaky':
        model = LeakyR2HandRilNet()
        model.load_state_dict(torch.load('r2handrilnet/weights/LeakyR2HandRilNet.pt'))
    elif net == 'bnorm':
        model = BnormR2HandRilNet()
        model.load_state_dict(torch.load('r2handrilnet/weights/BnormR2HandRilNet.pt'))
    else:
        model = LesserR2HandRilNet()
        model.load_state_dict(torch.load('r2handrilnet/weights/LesserR2HandRilNet.pt'))
    model.eval()
    image = image.convert('L')
    if invert:
        image = PIL.ImageOps.invert(image)
    image = image.resize((28, 28), Image.ANTIALIAS)
    image = np.array(image)
    image = np.expand_dims(image, axis=(0, 1))
    img_tensor = torch.tensor(image).float()
    with torch.no_grad():
        pred = model(img_tensor)
    prob = torch.nn.Softmax(dim=1)(pred)
    class_index = int(prob.argmax(dim=1))
    class_prob = float(prob[0][class_index])
    return class_index, class_prob
