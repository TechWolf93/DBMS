# =====================================
# Practical 2: Implementing Feedforward Neural Network using Keras & TensorFlow
# =====================================


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
%matplotlib inline


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print("Length of training set:", len(x_train))
print("Length of test set:", len(x_test))


print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)


plt.matshow(x_train[0])
plt.title("Example of MNIST Digit")
plt.show()


x_train = x_train / 255
x_test = x_test / 255


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model.summary()


model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=10)


test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss = %.3f" % test_loss)
print("Accuracy = %.3f" % test_acc)


n = random.randint(0, 9999)
plt.imshow(x_test[n])
plt.title("Random test image")
plt.show()


predicted_value = model.predict(x_test)
print("Handwritten number in the image is = %d" % np.argmax(predicted_value[n]))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


keras_model_path = '/content/keras_mnist_model'
model.save(keras_model_path)


restored_keras_model = tf.keras.models.load_model(keras_model_path)
print("Model saved and reloaded successfully.")
