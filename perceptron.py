import numpy as np
import math
from scipy import misc

gpath     = '/Users/atus/PycharmProjects/Deeplearning/model.ckpt'
ipath     = '/Users/atus/PycharmProjects/Deeplearning/img.png'
ppath     = '/Users/atus/PycharmProjects/Deeplearning/pattern.png'
normalize = lambda i: (255-i)/255

def display(arr):
    count = 0
    for i in arr:
        count += 1

        print '  ' if math.ceil(i) == 0.0 else '. ',
        if count % 28 == 0:
            print '\n'

def perceptron(x, w, bias):
    return np.dot(x, w) - bias > 0

img = misc.imread(ipath, True).flatten()
img = np.vectorize(normalize)(img)
display(img)

pattern = misc.imread(ppath, True).flatten()
pattern = np.vectorize(normalize)(pattern)
#display(pattern)

print '8' if perceptron(img, pattern, 25) else '0'