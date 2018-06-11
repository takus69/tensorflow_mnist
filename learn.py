import datetime

import numpy as np
import tensorflow as tf

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28 * 28))
x_train = np.true_divide(x_train, 255)
y_train = tf.keras.utils.to_categorical(y_train)

x_test = x_test.reshape((10000, 28 * 28))
x_test = np.true_divide(x_test, 255)
y_test = tf.keras.utils.to_categorical(y_test)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", input_shape=(28 * 28,)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learn
start = datetime.datetime.now()
print('{}: Start to train'.format(start))

model.fit(x_train, y_train, batch_size=100, epochs=200, verbose=2,
          validation_data=(x_test, y_test))

finish = datetime.datetime.now()
print('processing time: {}'.format(finish - start))
print('{}: Finish to train'.format(finish))

# Save model
model.save('mlp.h5')
