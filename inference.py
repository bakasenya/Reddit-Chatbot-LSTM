import tensorflow as tf
import pandas as pd
import logging
import numpy as np
import keras
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load weights and optimizer
train_.load_weights('weights.h5')
train_._make_train_function()

with open('optimizer.pkl', 'rb') as f:
    weight_values = pickle.load(f)
train_.optimizer.set_weights(weight_values)

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def decode_sequence(input_seq, tokenizer, model, max_length=16):
    """
    Decodes a given input sequence into a human-readable text sequence using a trained model and tokenizer.

    Parameters:
    - input_seq: The input sequence to decode.
    - tokenizer: The tokenizer used to process and convert sequences to text.
    - model: The trained model used for prediction.
    - max_length: Maximum length of the decoded sequence.

    Returns:
    - A decoded sequence as a string.
    """
    # Convert input sequence to token sequence and pad it
    input_seq = tokenizer.texts_to_sequences([input_seq])
    input_seq = pad_sequences(input_seq, max_length, padding='post')

    # Initialize target sequence (start with the 'startseq' token)
    target_seq = [[1]]  # 'startseq' token index
    target_seq = pad_sequences(target_seq, max_length, padding='post')

    stop_condition = False
    decoded_sentence = 'startseq'

    while not stop_condition:
        # Predict the next token based on the input and target sequence
        output_tokens = model.predict([input_seq, target_seq])

        # Get the token with the highest probability
        sampled_token_index = np.argmax(output_tokens[0, :])

        # Convert the token index to a character (word)
        sampled_char = tokenizer.sequences_to_texts([[sampled_token_index]])

        # Append the decoded character to the sentence
        decoded_sentence += ' ' + sampled_char[0]

        # Stop if 'endseq' is encountered or the length exceeds max_length
        if sampled_char[0] == 'endseq' or len(decoded_sentence.split(' ')) > max_length:
            stop_condition = True

        # Update target sequence for the next iteration
        target_seq = tokenizer.texts_to_sequences([decoded_sentence])
        target_seq = pad_sequences(target_seq, max_length, padding='post')

    # Return the decoded sentence without 'startseq' and 'endseq'
    return ' '.join(decoded_sentence.split(' ')[1:-1])
