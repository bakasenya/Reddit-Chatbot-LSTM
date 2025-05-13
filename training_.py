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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.client import device_lib
import random 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer




path='cleaned_data.csv'

def main():
    df=pd.read_csv(path,sep=';')
    df.drop('Unnamed: 0',axis=1,inplace = True)
    df['to']=df['to'].astype(str)
    df['from']=df['from'].astype(str)


    # In[50]:


    df['to']="startseq "+df['to']+' endseq'


    # In[52]:


    train,test=train_test_split(df,test_size=0.01)


    # In[53]:


    tokenizer=tf.keras.preprocessing.text.Tokenizer(
        num_words=10000,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True, split=' ', char_level=False, oov_token=None,
        document_count=0,
    )


    # In[54]:


    tokenizer.fit_on_texts(train['to'])
    decoder_input=tokenizer.texts_to_sequences(train['to'])
    encoder_input=tokenizer.texts_to_sequences(train['from'])
    decoder_input_test=tokenizer.texts_to_sequences(test['to'])
    encoder_input_test=tokenizer.texts_to_sequences(test['from'])


    # In[55]:




    # In[56]:


    encoder_input=pad_sequences(encoder_input,16,padding='post')
    encoder_input_test=pad_sequences(encoder_input_test,16,padding='post')


    # In[57]:


    


    # In[58]:


    encoder_input.shape


    # In[59]:


    
    print(device_lib.list_local_devices())


    # In[74]:


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
        return model


    # In[75]:


    train_ = define_models(16, 16, 128)
    adam=keras.optimizers.Adam()
    train_.compile(optimizer=adam, loss='categorical_crossentropy')
    print(train_.summary())


    # In[81]:



    
    def data_generator(output_seq, input_seq, tokenizer, max_length,batch=128):
        X1, X2, y = list(), list(), list() 
        n=0
        while 1:
            
            for key in range(0,len(input_seq)):
                key1=input_seq[key]  
                # encode the sequence
                seq = output_seq[key]
                
                # split one sequence into multiple X, y pairs
                for i in range(len(seq)-1):
                    if n==batch:
                        break
                    # split into input and output pair
                    in_seq, out_seq = seq[:i+1], seq[i+1]
                    if type(in_seq)!= type([]):
                        in_seq=list(in_seq)
                    # encode output sequence
                    out_seq = to_categorical(out_seq,10000,dtype='float16')
                        # store
                      
                    X1.append(key1)
                    X2.append(in_seq)
                    y.append(out_seq)
                    n+=1
                # yield the batch data
                    if n==(batch*10):
                        ran =random.sample(range(batch), int(batch))
                        n=0
                        yield [array(X1)[ran], pad_sequences(X2, maxlen=16,padding='post')[ran]], array(y)[ran]
                        
                        X1, X2, y = list(), list(), list()
                    
                    


    # In[143]:


    generator =data_generator(decoder_input, encoder_input, tokenizer, 16,128)
    validation_gen=data_generator(decoder_input_test, encoder_input_test, tokenizer, 16,1280)


    # In[ ]:


    train_.fit_generator(generator, epochs=10,steps_per_epoch=100000)


    # In[ ]:


    to_categorical(out_seq,10000,dtype='float16')


    # In[188]:


    def decode_sequence(input_seq,tokenizer,model):
        input_seq=tokenizer.texts_to_sequences([input_seq])
        input_seq=pad_sequences(input_seq,16,padding='post')
        target_seq=[[1]]
        target_seq=pad_sequences(target_seq,16,padding='post')

        stop_condition = False
        decoded_sentence = 'startseq'
        while not stop_condition:
            output_tokens = model.predict([input_seq, target_seq])
            sampled_token_index = np.argmax(output_tokens[0,:])
            sampled_char = tokenizer.sequences_to_texts([[sampled_token_index]])
            decoded_sentence += ' '+ sampled_char[0]

            if (sampled_char[0] == 'endseq' or
               len(decoded_sentence.split(' ')) > 16):
                break
            target_seq=tokenizer.texts_to_sequences([decoded_sentence])
            target_seq=pad_sequences(target_seq,16,padding='post')

        return ' '.join(decoded_sentence.split(' ')[1:-1])


    # In[244]:


    decode_sequence('why dont you say anything',tokenizer,train_)


    # In[253]:


    train_.save_weights('weights.h5')
    symbolic_weights = getattr(train_.optimizer, 'weights')
    weight_values = keras.backend.batch_get_value(symbolic_weights)
    with open('optimizer.pkl', 'wb') as f:
        pickle.dump(weight_values, f)




if __name__ == __main__:
    main()



