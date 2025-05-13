import tensorflow as tf
import pandas as pd
import logging
import numpy as np
import keras
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from keras.layers import TimeDistributed
import pickle
from keras import Input
from keras.preprocessing import sequence
from keras.layers import Embedding



def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=[n_input])
    emb=Embedding(n_output,256,input_length=n_input)
    encoder_emb=emb(encoder_inputs)
    encoder_emb=LSTM(n_units,return_sequences=True)(encoder_emb)
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_emb)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=[n_input])
    decoder_emb=emb(decoder_inputs)
    decoder_emb=LSTM(n_units,return_sequences=True)(decoder_emb)
    decoder_lstm = LSTM(n_units, return_sequences=False, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    train_ = define_models(16, 16, 128)
	adam=keras.optimizers.Adam()
	train_.compile(optimizer=adam, loss='categorical_crossentropy')
    return train_