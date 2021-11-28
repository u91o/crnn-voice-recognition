import tensorflow as tf

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, GRU, Reshape,
                                     Flatten, MaxPool2D)
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import config
import dataset


def crnn(input_shape, num_classes):
    """
    Defines the convolutional recurrent neural network model

    (input)
         v
      [convolution layer]
         v
        relu
         v
      [gru recurrent layer]
         v
      [gru recurrent layer]
         v
      [fully connected layer]
         v
        relu
         v
      [output layer]
         v
        softmax
    """
    model = Sequential()
    # reshape
    model.add((Reshape((config.height, config.width, 1),
                       input_shape=input_shape, name='conv_reshape')))
    # conv
    model.add(Conv2D(32, (20, 5), strides=(8,2), padding='same',
                     activation='relu', name='conv'))
    # reshape
    model.add(Reshape((32, 8*7), name='recurrent_reshape'))
    # recurrent
    model.add(GRU(32, return_sequences=True, name='recurrent1'))
    # recurrent
    model.add(GRU(32, name='recurrent2'))
    # fc
    model.add(Dense(64, activation='relu', name='fc'))
    model.add(Dense(num_classes, activation='softmax', name='output'))
    return model

def train(fn):
    """
    Training function for custom data
    """
    ds = dataset.Dataset()
    num_classes = len(ds.labels)
    train = ds.get_batches()

    model = fn((config.height, config.width), num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    callbacks = [
        ModelCheckpoint(filepath=config.checkpoint_dir),
        TensorBoard(log_dir=config.log_dir)
    ]
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    model.summary()
    
    model.fit(
        train,
        epochs=config.epochs,
        callbacks=callbacks)
    
    model.save(fn.__name__ + '.h5')

def tfds_train(fn):
    """
    Training function for TensorFlow Speech Commands Dataset
    """
    ds = dataset.TFDS()
    num_classes = len(ds.labels)
    train, validation, test = ds.get_batches()

    model = fn((config.height, config.width), num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    callbacks = [
        ModelCheckpoint(filepath=config.checkpoint_dir),
        TensorBoard(log_dir=config.log_dir)
    ]
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    model.summary()
    
    model.fit(
        train,
        epochs=config.epochs,
        validation_data=validation,
        callbacks=callbacks)

    print("\nTEST SET ACCURACY")
    model.evaluate(test)
    print('')

    model.save(fn.__name__ + '.h5')

def predict(model, samples):
    """
    Prediction function for custom dataset
    """
    model = load_model(model.__name__ + '.h5')
    assert model is not None
    # extract mfcc features
    mfccs,_ = dataset.process_samples(samples, None)
    # model takes batches, so we reshape the input to reflect batch size of 1
    mfccs = tf.reshape(mfccs, (1, config.height, config.width))
    
    scores = model.predict(mfccs)[0]
    labels = dataset.Dataset().labels

    # sort scores to get indices in decreasing order
    n_top = scores.argsort()[::-1]
    # # print all scores
    # for i in n_top:
    #     print("%s : %.2f" % (labels[i], scores[i]*100))

    # print top prediction
    i = n_top[0]
    print("%s : %.2f" % (labels[i], scores[i]*100))

def tfds_predict(model, samples):
    """
    Prediction function for TensorFlow dataset
    """
    model = load_model(model.__name__ + '.h5')
    assert model is not None
    # extract mfcc features
    mfccs,_ = dataset.process_samples(samples, None)
    # model takes batches, so we reshape the input to reflect batch size of 1
    mfccs = tf.reshape(mfccs, (1, config.height, config.width))
    
    scores = model.predict(mfccs)[0]
    labels = dataset.TFDS().labels

    # sort scores to get indices in decreasing order
    n_top = scores.argsort()[::-1]
    # # print all scores
    # for i in n_top:
    #     print("%s : %.2f" % (labels[i], scores[i]*100))

    # print top prediction
    i = n_top[0]
    print("%s : %.2f" % (labels[i], scores[i]*100))


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tfds_train(crnn)
