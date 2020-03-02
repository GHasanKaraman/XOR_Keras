import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint


# XOR Problem

x = np.array([[1,1],[1,0],[0,1],[0,0]])
y = np.array([[0],[1],[1],[0]])

epo = 50
batch = 2

# Creating Model

model = Sequential()

model.add(Dense(4, input_dim = 2))
model.add(Activation('relu'))
model.add(Dense(4, 'relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

callback = ModelCheckpoint('best_weights.hdf5', monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x, y, batch_size = batch, epochs = epo, verbose = 1, callbacks = [callback])


# Prediction
model.predict(np.array([1,1]).reshape(1,2))










