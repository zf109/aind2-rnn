import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras

from math import floor

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    # X = []
    # y = []

    # # reshape each 
    # X = np.asarray(X)
    # X.shape = (np.shape(X)[0:2])
    # y = np.asarray(y)
    # y.shape = (len(y),1)

    # return X,y
    """
        create an array using shifted list, then slice the array to create the X and y np.array
    """
    shifted_array = np.array([np.roll(series,-i)[:window_size+1] for i in range(len(series)-window_size)])
    return shifted_array[:,:window_size], shifted_array[:,window_size:window_size+1]


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(units=5, input_shape=(window_size, 1)))
    model.add(Dense(units=1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    valid_text = punctuation + list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    to_be_removed = ''.join(list(set(text).difference(valid_text)))
    # transtab = str.maketrans(None, to_be_removed)
    return text.translate(str.maketrans(to_be_removed, ''.join([' ']*len(to_be_removed))))


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    # X, y = window_transform_series(list(text), window_size)
    # inscope_rows = range(0, len(X), step_size)
    # inputs = np.array([X[i, :] for i in inscope_rows])
    # outputs = np.array([y[i, :] for i in inscope_rows])
    # num_datapoints = floor((len(text) - window_size)/step_size)
    # for i in range(num_datapoints):
        # step = i*step_size
        # inputs.append(text[step:step+window_size])
        # outputs.append(text[step+window_size])
    i = 0
    while i < len(text)-window_size:
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
        i += step_size
    return inputs, outputs


# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(units=200, input_shape=(window_size, num_chars)))
    model.add(Dense(units=num_chars, activation='linear'))
    model.add(Activation('softmax'))
    return model
