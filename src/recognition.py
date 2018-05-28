import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


class Recognition:
    def __init__(self, number_of_labels, dimension):
        self.model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        self.model.add(keras.layers.normalization.BatchNormalization(input_shape=(dimension, )))
        self.model.add(Dense(64, activation='relu', input_dim=dimension))
        self.model.add(Dropout(0.5))
        self.model.add(keras.layers.normalization.BatchNormalization())
        self.model.add(Dense(number_of_labels, activation='sigmoid'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    def train(self, data, labels):
        self.model.fit(data, labels, shuffle=True, epochs=20, batch_size=8)

    def test(self, data, labels):
        return self.model.evaluate(data, labels, batch_size=4)
