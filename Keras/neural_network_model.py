from keras import models
from keras import layers
from keras import optimizers



model = models.Sequential()
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dense(128, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

rmsproppy = optimizers.rmsprop(lr=0.01)
model.compile(optimizer=rmsproppy,
              loss='categorical_crossentropy',
              metrics=['accuracy'])