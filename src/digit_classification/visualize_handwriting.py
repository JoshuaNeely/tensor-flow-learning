from PIL import Image, ImageColor
import time
import climage


def visualize_handwriting_sample(handwriting_sample):
    handwriting_image = Image.fromarray(handwriting_sample, mode="L")
    handwriting_image.save(f'/tmp/handwriting.png')

    print('\n' * 15)
    print(climage.convert(f'/tmp/handwriting.png'))


def load_image_from_file(file_path):
    return Image.open(file_path)
