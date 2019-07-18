import words_to_sequence as wts
import read_clean_notes  as rcn
import neural_network_model as nnm
#Build training and test sets, set aside 10000 points to validate
x_val = wts.one_hot_results[:10000]
x_test = wts.one_hot_results[10000:20000]
x_train = wts.one_hot_results[20000:40000]

y_val = wts.one_hot_labels[:10000]
y_test = wts.one_hot_labels[10000:20000]
y_train = wts.one_hot_labels[20000:40000]

history = nnm.model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=128,
                    validation_data=(x_test, y_test))