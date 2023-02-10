import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow nvidia gpu warnings
import tensorflow as tf

from src.visualize_handwriting import visualize_handwriting_sample
from src.train_model import get_trained_model, get_predicted_classification

# the famous handwriting samples
# numbers 0-9, inputs 28x28 pixels
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def main():
    model = get_trained_model()

    for i in range(100):
        sample = x_train[i : i+1]
        truth_value = y_train[i : i+1]

        predicted_value = get_predicted_classification(model, sample)

        visualize_handwriting_sample(sample[0])

        print(f'predicted value:  {predicted_value}')
        print(f'truth value:  {truth_value[0]}')

        time.sleep(0.5)


main()
