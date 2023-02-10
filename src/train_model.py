import os
import numpy as np
np.set_printoptions(suppress=True, precision=4)

# silence tensorflow nvidia GPU warnings
# must be placed before import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def get_trained_model(retrain = False):
    model_file_path = '/home/jneely/git/tensor-flow/trained-model'

    if (retrain):
        print("retraining model")
        model = _train_model()
        model.save(model_file_path)

    else:
        try:
            print("loading trained model from disk")
            model = tf.keras.models.load_model(model_file_path)

        except OSError:
            print(f'no trained model found at {model_file_path}')
            print('retraining...')
            model = _train_model()
            model.save(model_file_path)

    return model


def get_predicted_classification(model, sample):
    #first_input = x_train[:1]
    first_input = sample[:1]
    raw_prediction = model(first_input).numpy()
    normalized_prediction = tf.nn.softmax(raw_prediction).numpy()
    classification_prediction = tf.math.argmax(normalized_prediction[0]).numpy()
    return classification_prediction


def _gather_mnist_training_data():
    # the famous handwriting samples
    # numbers 0-9, inputs 28x28 pixels
    mnist = tf.keras.datasets.mnist

    # x is input (28x28 hand writing), y is output (truth: 0-9 digit)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    return x_train, y_train


def _train_model():
    training_set_inputs, training_set_truth = _gather_mnist_training_data()

    # the final layer -Dense(10)- seems to be saying, I want 10 outputs (numbers 0-9)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # # 60 thousand samples, 28x28 pixels, a single per pixel
    # print(training_set_inputs.shape)
    # (60000, 28, 28, 1)

    # define a loss function, necessary for training
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # configure model with training strategy
    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    # train model with a number of "pit stops"
    model.fit(training_set_inputs, training_set_truth, epochs=5)

    return model


