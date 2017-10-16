import tensorflow as tf
import numpy as np
import operator
import math
from scipy import misc

gpath     = '/Users/atus/PycharmProjects/Deeplearning/model.ckpt'
ipath     = '/Users/atus/PycharmProjects/Deeplearning/img.png'
normalize = lambda i: (255-i)/255

img = misc.imread(ipath, True).flatten()
img = np.vectorize(normalize)(img)
count = 0
for i in img:
    count += 1

    print '  ' if math.ceil(i)==0.0 else '. ',
    if count%28 == 0:
        print '\n'


img = np.reshape(img, (1, 784))

np.reshape(img, (1, 784))

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

saver = tf.train.Saver()

sesh = tf.InteractiveSession()

saver.restore(sesh, gpath)

feed_dict = {x: img}
prediction = sesh.run(y, feed_dict)
index, value = max(enumerate(prediction[0]), key=operator.itemgetter(1))
print index

