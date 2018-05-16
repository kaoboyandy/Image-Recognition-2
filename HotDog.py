# settting the working directory and the path for the dataset 

from os.path import join

hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/hot_dog'

hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in 
                            ['1000288.jpg',
                             '127117.jpg']]

not_hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/not_hot_dog'
not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                            ['823536.jpg',
                             '99890.jpg']]

image_paths = hot_dog_paths + not_hot_dog_paths



# setting up the preprocessing functions and loading neccessary libraries

import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 224

def read_and_prep_images(image_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(image_path, target_size=(img_height, img_width)) for image_path in image_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)



# using the pre trained model to make the prediction

from tensorflow.python.keras.applications import ResNet50

my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
image_data = read_and_prep_images(image_paths)
my_preds = my_model.predict(image_data)


# loading the utility function for decoding prediction and visualize prediction

import sys

# Add directory holding utility functions to path to allow importing utility funcitons
sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils')
from decode_predictions import decode_predictions

# display the results with pictures and predictions

from IPython.display import Image, display

most_likely_labels = decode_predictions(my_preds, top=1, class_list_path='../input/resnet50/imagenet_class_index.json')
for i, image_path in enumerate(image_paths):
     display(Image(image_path))
     print(most_likely_labels[i])



