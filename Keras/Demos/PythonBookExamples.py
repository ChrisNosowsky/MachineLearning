# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:05:15 2019

@author: P01004LR
"""
#0 = Negative 1 = Positive
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import model_from_json



import numpy as np


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) #Means no word index will exceed 10k. Getting the 10,000 most frequent words




word_index = imdb.get_word_index()
reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()]) #Swap the key, value
decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in train_data[0]]) # i -3 is for the offset of the word index because of padding being reserved for 0, 1, and 2


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences): #Goes through all the #'s in each sequence and sets the zeros to ones.
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data) #Vectorizing trained data
x_test = vectorize_sequences(test_data) #Vecorizing test_data



y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32') #Vectorizing your labels into arrays with type float32

#Input data is vectors and the labels are scalars

#Hidden unit = a dimension in the representation space of the layer

#Makes sure that the dot product will project the input data on a 16-dimensional representation space
#Think of it as "How much freedom you're allowing the network to have when learning internal representations"
#More hidden units(a higher dimensional representation space) allows your network to learn more complex representations,
#but it makes the network more computationally expensive and may lead to learning unwanted patterns(improve performance on training data but not the test data)
#Relu acts as a way to zero out negative values
#Sigmoid squashes arbitrary values into the [0, 1] interval for probability purposes


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape = (10000,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

#Crossentropy measures the distance between probability distributions and predictions

x_val = x_train[:10000] #Setting aside a validation set
partial_x_train = x_train[10000:]


y_val = y_train[:10000] #Used a validation set to validate the train data actually
partial_y_train = y_train[10000:]

model.compile(optimizer=optimizers.RMSprop(lr=0.00008), loss='mean_squared_error', metrics=['acc']) #MSE Performed Better than binary_crossentropy
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data =(x_val, y_val))


##FOR Saving Model to JSON Below##

##Serialize model to JSON
#
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
#
##serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")
#



##2. Save Model Weights and Architecture Together
#This saves model weights, architecture, compilation details and optimizer state


#Saving - easy

scores = model.evaluate(x_val, y_val, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save("model.h5")
print("Saved model to disk")

#Plotting results 1. Training and validation loss. 2. Training and validation accuracy

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label = 'Training loss') #bo = blue dot
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') #b = solid blue line
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

##2
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()











