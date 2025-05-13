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
from keras.layers import LSTM, Dense, Input, Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random


path = 'cleaned_data.csv'


def main():
    # Load data
    df = pd.read_csv(path, sep=';')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df['to'] = df['to'].astype(str)
    df['from'] = df['from'].astype(str)

    # Add start and end tokens to the 'to' column
    df['to'] = "startseq " + df['to'] + ' endseq'

    # Split the data into train and test
    train, test = train_test_split(df, test_size=0.01)

    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=10000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', oov_token=None)
    
    # Fit tokenizer on training data
    tokenizer.fit_on_texts(train['to'])

    # Convert text to sequences
    decoder_input = tokenizer.texts_to_sequences(train['to'])
    encoder_input = tokenizer.texts_to_sequences(train['from'])
    decoder_input_test = tokenizer.texts_to_sequences(test['to'])
    encoder_input_test = tokenizer.texts_to_sequences(test['from'])

    # Pad sequences
    encoder_input = pad_sequences(encoder_input, 16, padding='post')
    encoder_input_test = pad_sequences(encoder_input_test, 16, padding='post')

    # Print the available devices
    print(tf.config.list_physical_devices())

    def define_models(n_input, n_output, n_units):
        # Define the encoder
        encoder_inputs = Input(shape=[n_input])
        emb = Embedding(n_output, 256, input_length=n_input)
        encoder_emb = emb(encoder_inputs)
        encoder_emb = LSTM(n_units, return_sequences=True)(encoder_emb)
        encoder = LSTM(n_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_emb)
        encoder_states = [state_h, state_c]

        # Define the decoder
        decoder_inputs = Input(shape=[n_input])
        decoder_emb = emb(decoder_inputs)
        decoder_emb = LSTM(n_units, return_sequences=True)(decoder_emb)
        decoder_lstm = LSTM(n_units, return_sequences=False, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)
        decoder_dense = Dense(n_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model

    # Define the model
    train_model = define_models(16, 16, 128)
    adam = keras.optimizers.Adam()
    train_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    print(train_model.summary())

    def data_generator(output_seq, input_seq, tokenizer, max_length, batch=128):
        X1, X2, y = list(), list(), list()
        n = 0
        while True:
            for key in range(0, len(input_seq)):
                key1 = input_seq[key]
                seq = output_seq[key]
                
                # Split one sequence into multiple X, y pairs
                for i in range(len(seq) - 1):
                    if n == batch:
                        break
                    in_seq, out_seq = seq[:i + 1], seq[i + 1]
                    in_seq = list(in_seq) if type(in_seq) != list else in_seq
                    out_seq = to_categorical(out_seq, 10000, dtype='float16')
                    
                    # Store the data
                    X1.append(key1)
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1

                if n == (batch * 10):
                    ran = random.sample(range(batch), int(batch))
                    n = 0
                    yield [array(X1)[ran], pad_sequences(X2, maxlen=16, padding='post')[ran]], array(y)[ran]
                    
                    X1, X2, y = list(), list(), list()

    # Data generators
    generator = data_generator(decoder_input, encoder_input, tokenizer, 16, 128)
    validation_gen = data_generator(decoder_input_test, encoder_input_test, tokenizer, 16, 1280)

    # Train the model
    train_model.fit(generator, epochs=10, steps_per_epoch=100000, validation_data=validation_gen)

    # Save weights and optimizer state
    train_model.save_weights('weights.h5')
    symbolic_weights = getattr(train_model.optimizer, 'weights')
    weight_values = keras.backend.batch_get_value(symbolic_weights)
    with open('optimizer.pkl', 'wb') as f:
        pickle.dump(weight_values, f)


# Call main function
if __name__ == '__main__':
    main()
