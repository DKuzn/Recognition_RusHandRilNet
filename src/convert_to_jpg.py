from PIL import Image

ALPHABET = 'а б в г д е ё ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я'.split()
PATH_S = '../Dataset_png/'
PATH_D = '../Dataset/'

for i in range(1, len(ALPHABET) + 1):
    for j in range(1, 431):
        if i < 10:
            path_s = PATH_S + ALPHABET[i - 1] + '/0' + str(i) + '_'
            path_d = PATH_D + ALPHABET[i - 1] + '/0' + str(i) + '_'
        else:
            path_s = PATH_S + ALPHABET[i - 1] + '/' + str(i) + '_'
            path_d = PATH_D + ALPHABET[i - 1] + '/' + str(i) + '_'

        if j < 10:
            path_s += '0' + str(j) + '.png'
            path_d += '0' + str(j) + '.jpg'
        else:
            path_s += str(j) + '.png'
            path_d += str(j) + '.jpg'

        img = Image.open(path_s)
        rgb_img = img.convert('RGB')
        rgb_img.save(path_d, quality=100)
