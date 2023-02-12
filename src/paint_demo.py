import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow nvidia gpu warnings

import sys
import time
import numpy as np

import tensorflow as tf

from src import train_model
from src.visualize_handwriting import visualize_handwriting_sample, load_image_from_file
from src.paint import launch_paint_loop


def demo():
    model = train_model.get_trained_model(False)

    def save_callback(filepath):
        img = load_image_from_file(filepath)
        a = np.array(img)
        a = a[:, :, 0]  # all three chanels are identical, in this case
        a = a[np.newaxis, :, :]

        sample = a
        predicted_value = train_model.get_predicted_classification(model, sample)
        all_predictions = train_model.get_all_classifications(model, sample)

        #visualize_handwriting_sample(sample[0])

        print(f'predicted value:  {predicted_value}')
        print(all_predictions)

    launch_paint_loop(save_callback)
