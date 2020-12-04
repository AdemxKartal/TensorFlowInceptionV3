

import tensorflow as tf


NUM_CLASSES = 102
image_height = 299
image_width = 299
channels = 3

IMAGE_RES = 229


def get_model():
    model = tf.keras.applications.InceptionV3(include_top=True,weights=None,classes=NUM_CLASSES)
    model.build(input_shape=(None, image_height, image_width, channels))
    model.summary() #prints model information
    return model


def get_loss_object():
    return tf.keras.losses.SparseCategoricalCrossentropy()
    #return  tf.keras.losses.CategoricalCrossentropy

def get_optimizer():
    return tf.keras.optimizers.Adadelta()

def get_train_loss():
    return tf.keras.metrics.Mean(name='train_loss')

def get_train_accuracy():
    return tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

def get_valid_loss():
    return tf.keras.metrics.Mean(name='valid_loss')

def get_valid_accuracy():
    return tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
