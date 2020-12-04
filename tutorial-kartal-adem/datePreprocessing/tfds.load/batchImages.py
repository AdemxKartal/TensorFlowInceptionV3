

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import tensorflow as tf


IMAGE_RES = 229
batch_size=32
def resize_image(image,label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label



if __name__ == '__main__':
    dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
    dataset_info
    test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']
    training_set=training_set.map(resize_image)

    get_label_name = dataset_info.features['label'].int2str
    training_batch = training_set.batch(batch_size)


    for image,label in training_batch:
        print('label: ', label)



