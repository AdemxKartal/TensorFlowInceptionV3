

import tensorflow as tf
import tensorflow_datasets as tfds


ImageSize = 229
BatchSize=32


def get_batchSize():
    return BatchSize
def get_labelName(dataSetInfo):
    labelName= dataSetInfo.features['label'].int2str
    return labelName


def get_dataset():
    dataset, dataSetInfo = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
    dataSetInfo
    testSet, trainingSet, validationSet = dataset['test'], dataset['train'], dataset['validation']
    return testSet,trainingSet,validationSet,dataSetInfo

def resize_image(image,label):
    image = tf.image.resize(image, (ImageSize, ImageSize))/255.0
    return image, label

def map_dataset(dataset):
    mappedDataset=dataset.map(resize_image)
    return mappedDataset


def batch_dataset(mappedDataset):
    batchedDataSet = mappedDataset.batch(BatchSize)
    return batchedDataSet

def get_numberOfBatchExample(batch):
    numberOfBatchExample = 0
    for example in batch:
        numberOfBatchExample=numberOfBatchExample+1
    return numberOfBatchExample






