# coding: utf-8

import numpy as np
import random
import string
from wheezy.captcha.image import warp, rotate, offset, background, text, curve, noise, smooth, captcha
from skimage.io import imshow
from sklearn.model_selection import train_test_split

# Captcha Generator Settings
fonts=['./comic-sans-ms.ttf']
text_drawings = [
                #warp(dx_factor=0.27, dy_factor=0.21),
                rotate(angle=25),
                offset(dx_factor=0.1, dy_factor=0.2),
                ]
drawings=[
         background(color='#dcdcdc'),
         text(fonts=fonts, font_sizes=(28, 28, 28), drawings=text_drawings, color='#32fa32', squeeze_factor=1),
         #curve(),
         noise(number=60, color='#69d748', level=2),
         smooth(),
         ]

generator = captcha(drawings=drawings, width=150, height=50)

def captcha_gen(string_size, generator):
    string = chars_generator(string_size)
    img = generator(string)
    pix = np.array(img)
    return pix, string

def chars_generator(size=4, chars=string.digits + string.ascii_uppercase):
    return ''.join(random.choice(chars) for _ in range(size))

def data_gen(size=10000, captcha_width=150, captcha_height=50, string_size=4):
    char_set = list(string.digits + string.ascii_uppercase)
    char_set_size = len(char_set)

    X = np.zeros(shape=(size,captcha_height,captcha_width,3), dtype='uint8')
    y = np.zeros(shape=(size, string_size, char_set_size))

    for i in range(size):
        X[i], string_ = captcha_gen(string_size, generator)
        for j, char in enumerate(string_):
            k = char_set.index(char)
            y[i,j,k] += 1

    X = X.astype('float32')

    return X, y

if __name__ == '__main__':
    X, y = data_gen(size=10000)

    X[0].shape, y[0].shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train /= 255
    X_test /= 255

    np.save('data/train_inputs',X_train)
    np.save('data/train_targets',y_train)

    np.save('data/test_inputs',X_test)
    np.save('data/test_targets',y_test)
