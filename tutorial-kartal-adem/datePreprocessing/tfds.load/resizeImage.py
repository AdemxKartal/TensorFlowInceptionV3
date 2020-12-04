

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import matplotlib.pyplot as plt
IMAGE_RES = 229
import tensorflow as tf

def format_image(image,label):
    image = tf.image.resize(image, (IMAGE_RES,IMAGE_RES))/255.0
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label

def resize_image(image,label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label

if __name__ == '__main__':
    dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
    dataset_info
    test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']

    test_set = test_set.map(resize_image)
    training_set=training_set.map(resize_image)
    validation_set=validation_set.map(resize_image)
    get_label_name = dataset_info.features['label'].int2str
    image, label = next(iter(training_set))
    _ = plt.imshow(image)
    _ = plt.title(get_label_name(label))
    plt.show()



