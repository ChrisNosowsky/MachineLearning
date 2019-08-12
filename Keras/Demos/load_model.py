# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:41:23 2019

@author: P01004LR
"""

#import PythonBookExamples as pb
import numpy as np
from keras.datasets import imdb
from keras.models import load_model

##load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
##Load weights into new model
#loaded_model = model_from_json(loaded_model_json)
#print("Loaded model from disk")
#
## Evaluate loaded model on test data
#loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['acc'])
#score = loaded_model.evaluate(partial_x_train, partial_y_train, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



#Loading
model = load_model('model.h5')
#summarize model
model.summary()
#load dataset
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences): #Goes through all the #'s in each sequence and sets the zeros to ones.
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) #Means no word index will exceed 10k. Getting the 10,000 most frequent words
x_train = vectorize_sequences(train_data) #Vectorizing trained data
y_train = np.asarray(train_labels).astype('float32')
x_val = x_train[:10000] #Setting aside a validation set
y_val = y_train[:10000] #Used a validation set to validate the train data actually


score = model.evaluate(x_val, y_val, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))




