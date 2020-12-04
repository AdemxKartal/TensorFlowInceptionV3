from dataPreprocessing import get_dataset, map_dataset, batch_dataset, get_numberOfBatchExample, get_batchSize
from defineNetwork import get_model, get_optimizer, get_valid_loss,get_valid_accuracy,get_train_loss,get_loss_object,\
    get_train_accuracy,get_valid_loss_object

import tensorflow as tf
import math
import os

save_model_dir = os.getcwd()
EPOCHS = 2
BatchSize = get_batchSize()

if __name__ == '__main__':
    TestSet,TrainingSet,ValidationSet,DataSetInfo=get_dataset()
    TestSet=map_dataset(TestSet)
    TrainingSet=map_dataset(TrainingSet)
    ValidationSet=map_dataset(ValidationSet)
    TrainingSetNumber= get_numberOfBatchExample(TrainingSet)
    TestBatch = batch_dataset(TestSet)
    TrainingBatch=batch_dataset(TrainingSet)
    ValidationBatch=batch_dataset(ValidationSet)


    inceptionNetwork= get_model()
    loss_object=get_loss_object()
    optimizer=get_optimizer()

    train_loss=get_train_loss()
    train_accuracy=get_train_accuracy()

    valid_loss=get_valid_loss()
    valid_accuracy=get_valid_accuracy()
    valid_loss_object = get_valid_loss_object()
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:  # tf.GradientTape: Record operations for automatic differentiation
            predictions = inceptionNetwork(images, training=True)
            loss_aux = loss_object(y_true=labels, y_pred=predictions)
            loss = 0.5 * loss_aux + 0.5 * loss_object(y_true=labels, y_pred=predictions)
            #loss_aux = loss_object(labels, predictions)
            #loss = 0.5 * loss_aux + 0.5 * loss_object(labels, predictions)
        gradients = tape.gradient(loss, inceptionNetwork.trainable_variables)  # Get gradients of loss wrt the weights.
        optimizer.apply_gradients(grads_and_vars=zip(gradients, inceptionNetwork.trainable_variables))  # Update the weights of the model.

        train_loss(loss)
        train_accuracy(labels, predictions)



    @tf.function
    def valid_step(images, labels):
        predictions = inceptionNetwork(images, training=False)
        #v_loss = loss_object(y_true=labels, y_pred=predictions)
        v_loss = valid_loss_object(labels, predictions)
        valid_loss(v_loss)
        #valid_accuracy(labels, predictions)


    # start training
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0

        for valid_image, valid_label in ValidationBatch:
            valid_step(valid_image, valid_image)

        #print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
            #  "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
             #                                                     EPOCHS,
             #                                                     train_loss.result(),
             #                                                     train_accuracy.result(),
              #                                                    valid_loss.result(),
               #                                                   valid_accuracy.result()))

    inceptionNetwork.save_weights(filepath=save_model_dir, save_format='tf')

