# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:14:14 2019

@author: P01004LR
"""



from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)



word_index = reuters.get_word_index()
reverse_word_index = dict([value, key] for (key, value) in word_index.items())
decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#One hot encoding

# =============================================================================
# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1
#     return results
# 
# one_hot_train_labels = to_one_hot(train_labels) #Vectorized training labels
# one_hot_test_labels = to_one_hot(test_labels) #Vectorized test labels
# 
# =============================================================================

#Or the built in way works too

from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


#Could also not do one hot, but cast them as tensors
#y_train = np.array(train_labels)
#y_test = np.array(test_labels)
#This would make it so that you can't use categorical_crossentropy as your loss function, but instead sparse_categorical_crossentropy

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax')) #Output layer


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Gotta set aside a validation set

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]




history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size = 512,
                    validation_data=(x_val, y_val))

#Plotting
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Plotting training and validation accuracy
plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#Generating predictions on new data

predictions = model.predict(x_test)



#Don't have a low dimensional intermediate layer(low hidden layers)
#If you do, you will be trying to compress a lot of information into an intermediate space that is too low-dimensional
#Can cram mostt of the necessary information into these dimensions, just not all of it




























