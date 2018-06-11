import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass
from sklearn import metrics
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('mlp.h5')

# Predict raw data
images = []

for i in range(10):
    im = Image.open('data/{}.png'.format(i)).convert('L')
    im = np.array(im)
    im = im.reshape((28 * 28))
    im = np.true_divide(im, 255)
    images.append(im)
pred = model.predict_classes(np.array(images))
score = metrics.accuracy_score([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pred)

print(pred)
print('Accuracy is {}%'.format(score * 100))


# Define preprocess
def preprocess(image):
    # Resize to 20 * 20
    im = image.resize((20, 20))

    # Pasete to 28 * 28
    im_ret = Image.new('L', (28, 28))
    im_ret.paste(im, (4, 4))

    # Compute the center of mass and translate the image
    # to the point at the center of 28 * 28
    y_center, x_center = center_of_mass(np.array(im_ret))
    x_move = x_center - 14
    y_move = y_center - 14
    im_ret = im_ret.transform(size=(28, 28), method=Image.AFFINE,
                              data=(1, 0, x_move, 0, 1, y_move))
    return im_ret


# Predict proprocessed data
images = []

for i in range(10):
    im = Image.open('data/{}.png'.format(i)).convert('L')
    im = preprocess(im)
    im.save('data/conv{}.png'.format(i))
    im = np.array(im)
    im = im.reshape((28 * 28))
    im = np.true_divide(im, 255)
    images.append(im)
pred = model.predict_classes(np.array(images))
score = metrics.accuracy_score([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pred)

print(pred)
print('Accuracy is {}%'.format(score * 100))

# Predict shifted data
images = []

for i in ['', 't', 'b', 'l', 'r']:
    im = Image.open('data/conv7{}.png'.format(i)).convert('L')
    im = np.array(im)
    im = im.reshape((28 * 28))
    im = np.true_divide(im, 255)
    images.append(im)
pred = model.predict_classes(np.array(images))
score = metrics.accuracy_score([7, 7, 7, 7, 7], pred)

print(pred)
print('Accuracy is {}%'.format(score * 100))
