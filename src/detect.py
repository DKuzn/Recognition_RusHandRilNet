import torch
import numpy as np
from R2HandRilNet import R2HandRilNet
from PIL import Image
import PIL.ImageOps

model = R2HandRilNet()
model.load_state_dict(torch.load('weights/R2HandRilNet.pt'))
model.eval()


def detect_letter(image):
    image.convert('L')
    image = PIL.ImageOps.invert(image)
    image = image.resize((28, 28), Image.ANTIALIAS)
    image = np.array(image)
    image = np.expand_dims(image, axis=(0, 1))
    img_tensor = torch.tensor(image).float()
    pred = model(img_tensor)
    prob = torch.nn.Softmax(dim=1)(pred)
    class_index = int(prob.argmax(dim=1))
    class_prob = float(prob[0][class_index])
    return class_index, class_prob
