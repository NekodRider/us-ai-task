import scipy.io
import scipy.misc
import numpy as np


def load_image(path):
    image = scipy.misc.imread(path)
    image = image / 255.0
    image = np.reshape(image, ((1,) + image.shape))
    return image


def save_image(image, path):
    img = image
    img = img[0] * 255
    img = np.clip(img, 0, 255).astype('uint8')
    scipy.misc.imsave(path, img)
