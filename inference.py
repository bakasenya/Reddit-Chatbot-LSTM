import tensorflow as tf
import pandas as pd
import logging
import numpy as np
import keras


train_.load_weights('weights.h5')
train_._make_train_function()
with open('optimizer.pkl', 'rb') as f:
    weight_values = pickle.load(f)
train_.optimizer.set_weights(weight_values)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def decode_sequence(input_seq,tokenizer,model,length=16):
    input_seq=tokenizer.texts_to_sequences([input_seq])
    input_seq=pad_sequences(input_seq,length,padding='post')
    target_seq=[[1]]
    target_seq=pad_sequences(target_seq,length,padding='post')

    stop_condition = False
    decoded_sentence = 'startseq'
    while not stop_condition:
        output_tokens = model.predict([input_seq, target_seq])
        sampled_token_index = np.argmax(output_tokens[0,:])
        sampled_char = tokenizer.sequences_to_texts([[sampled_token_index]])
        decoded_sentence += ' '+ sampled_char[0]

        if (sampled_char[0] == 'endseq' or
           len(decoded_sentence.split(' ')) > length):
            break
        target_seq=tokenizer.texts_to_sequences([decoded_sentence])
        target_seq=pad_sequences(target_seq,length,padding='post')

    return ' '.join(decoded_sentence.split(' ')[1:-1])