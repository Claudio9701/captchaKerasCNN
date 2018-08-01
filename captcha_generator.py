import math
import captcha
import matplotlib.pyplot as plt

from wheezy.captcha.image import captcha

from wheezy.captcha.image import background
from wheezy.captcha.image import curve
from wheezy.captcha.image import noise
from wheezy.captcha.image import smooth
from wheezy.captcha.image import text

from wheezy.captcha.image import offset
from wheezy.captcha.image import rotate
from wheezy.captcha.image import warp

import random
import string
captcha_image = captcha(drawings=[background(color='whitesmoke'), text(fonts=['/Users/cortega/Documents/arcanine/comic-sans-ms.ttf'], drawings=[warp(),rotate(),offset()]), noise(),smooth()])

image = captcha_image(random.sample(string.ascii_uppercase + string.digits, 4))
plt.imsave('test.png', image)
