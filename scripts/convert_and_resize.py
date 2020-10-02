from PIL import Image
import os

ALPHABET = 'а б в г д е ё ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я'.upper().split()
PATH_S = '../CoMNIST/'
PATH_D = '../CoMNIST32_jpg/'

for i in range(34, 67):
    for j in range(1, len(os.listdir(PATH_S + ALPHABET[i - 34])) + 1):
        if i < 10:
            path_s = PATH_S + ALPHABET[i - 34] + '/0' + str(i) + '_'
            path_d = PATH_D + ALPHABET[i - 34] + '/0' + str(i) + '_'
        else:
            path_s = PATH_S + ALPHABET[i - 34] + '/' + str(i) + '_'
            path_d = PATH_D + ALPHABET[i - 34] + '/' + str(i) + '_'

        if j < 10:
            path_s += '0' + str(j) + '.png'
            path_d += '0' + str(j) + '.jpg'
        else:
            path_s += str(j) + '.png'
            path_d += str(j) + '.jpg'

        png = Image.open(path_s)
        png.load()
        background = Image.new('RGB', png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])
        background.thumbnail((32, 32), Image.ANTIALIAS)
        background.save(path_d, 'JPEG', quality=100)
