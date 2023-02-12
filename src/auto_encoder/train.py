import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.models import Model

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  # this is used implicitly somewhere in the training process...
  # part of the Model interface?
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def _gather_dataset():
  (x_train, _), (x_test, _) = mnist.load_data()
  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  return x_train, x_test

def _train_autoencoder():
  x_train, x_test = _gather_dataset()
  latent_dimensions = 80
  epochs = 1

  autoencoder = Autoencoder(latent_dimensions)
  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
  autoencoder.fit(x_train, x_train,
                  epochs=epochs,
                  shuffle=True,
                  validation_data=(x_test, x_test))
  return autoencoder

def get_trained_model(retrain = False):
    model_file_path = '/tmp/trained-autoencoder'

    if (retrain):
        print("retraining model")
        model = _train_autoencoder()
        model.save(model_file_path)

    else:
        try:
            print("loading trained model from disk")
            model = tf.keras.models.load_model(model_file_path)

        except OSError:
            print(f'no trained model found at {model_file_path}')
            print('retraining...')
            model = _train_autoencoder()
            model.save(model_file_path)

    return model

def demo():
  x_train, x_test = _gather_dataset()
  autoencoder = get_trained_model(False)

  encoded_imgs = autoencoder.encoder(x_test).numpy()
  print(encoded_imgs.shape)
  decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

  # n = 10
  # plt.figure(figsize=(20, 4))
  # for i in range(n):
  #   # display original
  #   ax = plt.subplot(2, n, i + 1)
  #   plt.imshow(x_test[i])
  #   plt.title("original")
  #   plt.gray()
  #   ax.get_xaxis().set_visible(False)
  #   ax.get_yaxis().set_visible(False)

  #   # display reconstruction
  #   ax = plt.subplot(2, n, i + 1 + n)
  #   plt.imshow(decoded_imgs[i])
  #   plt.title("reconstructed")
  #   plt.gray()
  #   ax.get_xaxis().set_visible(False)
  #   ax.get_yaxis().set_visible(False)
  # plt.show()
