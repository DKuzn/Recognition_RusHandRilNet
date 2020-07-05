import cv2 as cv
from itertools import groupby


def view_image(image, name_of_window):
    cv.namedWindow(name_of_window, cv.WINDOW_NORMAL)
    cv.imshow(name_of_window, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def image_binarisation(image):
    blur = cv.medianBlur(image, 3)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    ret, binary_image = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return binary_image


def separate(binary_img):
    letter_tres = []
    white_pixels = []
    for j in range(0, binary_img.shape[1], 20):
        for i in range(binary_img.shape[0]):
            if binary_img[i][j] == 255:
                white_pixels.append((i, j))
        if white_pixels:
            letter_tres.append(len(white_pixels))
            white_pixels.clear()
    return letter_tres


def find_minimum_weights(weights):
    min_values = []
    sort = sorted(weights)
    weights_values = [el for el, _ in groupby(sort)]
    min_values.append(weights_values[0])
    min_values.append(weights_values[1])
    if 2 <= weights_values[2] - weights_values[1] < 5:
        min_values.append(weights_values[2])
    if 2 <= weights_values[3] - weights_values[2] < 5:
        min_values.append(weights_values[3])
    return min_values

def find_letters_edges(img):
    bin = image_binarisation(img)
    weights = separate(bin)
    print(weights)
    indexes = []
    edges = []
    for i in range(0, bin.shape[1], 20):
        indexes.append(i)
    mins = find_minimum_weights(weights)
    print(mins)
    for i in range(len(weights)):
        if weights[i] in mins:
            edges.append(indexes[i])
    for i in range(len(edges)):
        x3 = edges[i]
        y3 = 0
        x4 = edges[i]
        y4 = 200
        cv.line(img, (x3, y3), (x4, y4), (255, 0, 0), 1)
    view_image(img, 'img')


img = cv.imread('crop8.jpg')
find_letters_edges(img)