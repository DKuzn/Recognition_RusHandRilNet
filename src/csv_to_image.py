import pandas as pd
import numpy as np
import cv2 as cv


PATH = '/home/dmitry/Загрузки/HMCC/HMCC_images/'

csv_image = pd.read_csv('/home/dmitry/Загрузки/HMCC/HMCC_balanced.csv', encoding='UTF-8')

for i in range(len(csv_image)):
    array = np.array(csv_image.loc[i])
    path = PATH + str(array[0]) + '/' + str(i) + '.jpg'
    img_array = np.uint8(array[1:])
    img = img_array.reshape(28, 28)
    cv.imwrite(path, img)