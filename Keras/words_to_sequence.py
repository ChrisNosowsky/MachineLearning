from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

import read_clean_notes as rcn
docs = []
for note in rcn.notes:
    docs.append(note[1])
labels = []
for note in rcn.notes:
    note_cat_int = 100
    if note[2] == 'CA':
        note_cat_int = 0
    elif note[2] == 'GL':
        note_cat_int = 1
    elif note[2] == 'WC':
        note_cat_int = 2
    labels.append(note_cat_int)

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(docs)
sequences = tokenizer.texts_to_sequences(docs)
#result = []
#for text in docs:
#    word_count = len(text)
#    result.append(hashing_trick(text, round(word_count*1.3), hash_function='md5'))

#result = hashing_trick(docs, round(sequence_size*1.3), hash_function='md5')
one_hot_results = tokenizer.texts_to_matrix(docs, mode='binary')
#tokenizer_labels = Tokenizer(num_words=4)
#tokenizer_labels.fit_on_texts(labels)
#sequences_labels = tokenizer_labels.texts_to_sequences(labels)
one_hot_labels = to_categorical(labels,3)
#word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))
