from dataPreprocessing import get_dataset, map_dataset, batch_dataset, get_numberOfBatchExample, get_batchSize
from defineNetwork import get_model, get_optimizer, get_valid_loss,get_valid_accuracy,get_train_loss,get_loss_object,\
    get_train_accuracy

import tensorflow as tf
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
    counter = 0

    inceptionNetwork.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=optimizer, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    inceptionNetwork.fit(TrainingBatch, epochs = 1, validation_data=ValidationBatch)
