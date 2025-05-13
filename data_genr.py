import tensorflow as tf
import pandas as pd
import pickle
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

# Tokenizer function to preprocess text data
def tokenizer(file_path):
    # Read the dataset
    df = pd.read_csv(file_path, sep=';')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Prepare 'from' and 'to' columns for processing
    df['to'] = df['to'].astype(str)
    df['from'] = df['from'].astype(str)
    df['to'] = "startseq " + df['to'] + ' endseq'
    
    # Split dataset into training and test sets
    train, test = train_test_split(df, test_size=0.01)

    # Initialize the tokenizer with the specified parameters
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=10000,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=' ',
        char_level=False,
        oov_token=None,
        document_count=0
    )
    
    # Fit tokenizer on the 'to' column of the training set
    tokenizer.fit_on_texts(train['to'])
    
    # Convert text sequences to integer sequences
    decoder_input = tokenizer.texts_to_sequences(train['to'])
    encoder_input = tokenizer.texts_to_sequences(train['from'])
    decoder_input_test = tokenizer.texts_to_sequences(test['to'])
    encoder_input_test = tokenizer.texts_to_sequences(test['from'])

    # Pad sequences to ensure uniform length
    encoder_input = pad_sequences(encoder_input, 16, padding='post')
    encoder_input_test = pad_sequences(encoder_input_test, 16, padding='post')
    
    # Save tokenizer for later use
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return decoder_input, encoder_input, decoder_input_test, encoder_input_test, tokenizer

# Data generator function for batching the data during training
def data_generator(output_seq, input_seq, tokenizer, max_length, batch_size=128):
    batch_size *= 10  # Scale the batch size by a factor of 10
    X1, X2, y = [], [], []  # Initialize the input and output lists
    n = 0  # Initialize the counter for batch size
    
    while True:
        for key in range(len(input_seq)):
            key_input = input_seq[key]
            seq = output_seq[key]  # Get the corresponding output sequence
            
            # Split one sequence into multiple X, y pairs
            for i in range(len(seq) - 1):
                if n == batch_size:
                    break

                # Split the sequence into input and output pair
                in_seq, out_seq = seq[:i + 1], seq[i + 1]
                if type(in_seq) != list:
                    in_seq = list(in_seq)

                # Encode the output sequence as one-hot
                out_seq = to_categorical(out_seq, 10000, dtype='float16')

                # Append the data to the lists
                X1.append(key_input)
                X2.append(in_seq)
                y.append(out_seq)
                n += 1
            
            # Yield the batch data when it reaches the batch size
            if n == batch_size // 10:
                selected_indices = random.sample(range(batch_size), batch_size // 10)
                n = 0
                yield [np.array(X1)[selected_indices], pad_sequences(X2, maxlen=16, padding='post')[selected_indices]], np.array(y)[selected_indices]

                # Reset the lists for the next batch
                X1, X2, y = [], [], []
