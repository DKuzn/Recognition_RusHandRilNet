import csv
import numpy as np
import cv2 as cv
import os
import tqdm
import argparse

dataset_dir = '../Dataset'
train_dir = '../Dataset/Train'
test_dir = '../Dataset/Test'


def get_dataset(csv_path: str):
    labels_open = open('labels', 'r')
    labels_open = labels_open.read().split('\n')
    labels_open = [i.split(' ') for i in labels_open]
    labels = {}

    for i in labels_open:
        labels[i[0]] = i[1]

    dataset = {}
    for j in range(66):
        if not labels[str(j)] == 'invalid':
            key = labels[str(j)]
            dataset[key] = []

    csv_images = open(csv_path, 'r')
    reader = csv.reader(csv_images)

    for row in tqdm.tqdm(reader):
        key = labels[str(row[0])]
        if not key == 'invalid':
            dataset[key].append(row[1:])

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for k in range(66):
        folder = labels[str(k)]
        if not folder == 'invalid':
            train_path = train_dir + '/' + folder
            test_path = test_dir + '/' + folder
            if not os.path.exists(train_path):
                os.mkdir(train_path)
            if not os.path.exists(test_path):
                os.mkdir(test_path)

    for letter in tqdm.tqdm(range(66)):
        key = labels[str(letter)]
        if not key == 'invalid':
            for file in range(1, 5251, 1):
                if file <= 3750:
                    path = train_dir + '/' + key + '/' + str(file) + '.jpg'
                else:
                    path = test_dir + '/' + key + '/' + str(file) + '.jpg'
                img_array = dataset[key][file - 1]
                np_img = np.array(img_array)
                np_img = np.uint8(np_img)
                img = np_img.reshape(28, 28)
                cv.imwrite(path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to extract dataset images from .csv')
    parser.add_argument('path_to_csv', metavar='path_to_csv', type=str,
                        help='Path to csv with grayscale images for dataset.')
    args = parser.parse_args()
    try:
        path_to_csv = args.path_to_csv
        get_dataset(path_to_csv)
    except FileNotFoundError:
        print('Path is incorrect.')
