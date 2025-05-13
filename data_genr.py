import tensorflow as tf
import pandas as pd
import logging
import numpy as np
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random 





def tokenizer(path):
	df=pd.read_csv(path,sep=';')
	df.drop('Unnamed: 0',axis=1,inplace = True)	
	df['to']=df['to'].astype(str)
	df['from']=df['from'].astype(str)
	df['to']="startseq "+df['to']+' endseq'
	train,test=train_test_split(df,test_size=0.01)
	tokenizer=tf.keras.preprocessing.text.Tokenizer(
    	num_words=10000,
    	filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    	lower=True, split=' ', char_level=False, oov_token=None,
    	document_count=0)
    tokenizer.fit_on_texts(train['to'])
	decoder_input=tokenizer.texts_to_sequences(train['to'])
	encoder_input=tokenizer.texts_to_sequences(train['from'])
	decoder_input_test=tokenizer.texts_to_sequences(test['to'])
	encoder_input_test=tokenizer.texts_to_sequences(test['from'])
	encoder_input=pad_sequences(encoder_input,16,padding='post')
	encoder_input_test=pad_sequences(encoder_input_test,16,padding='post')
	# saving
	with open('tokenizer.pickle', 'wb') as handle:
    	pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return decoder_input,encoder_input,decoder_input_test,encoder_input_test,tokenizer



def data_generator(output_seq, input_seq, tokenizer, max_length,batch=128):
	batch=batch*10
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
            # yield the batch data and randomly select batch size from size of batch*10
                if n==(int(batch/10)):
                    ran =random.sample(range(batch), int(batch/10))
                    n=0
                    yield [array(X1)[ran], pad_sequences(X2, maxlen=16,padding='post')[ran]], array(y)[ran]
                    
                    X1, X2, y = list(), list(), list()
                
                



