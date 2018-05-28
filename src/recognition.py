from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization


class Recognition:
    def __init__(self, number_of_labels, dimension):
        self.model = Sequential()

        self.model.add(BatchNormalization(input_shape=(dimension, )))
        self.model.add(Dense(64, activation='relu', input_dim=dimension))

        self.model.add(BatchNormalization())
        self.model.add(Dense(number_of_labels, activation='sigmoid'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    def train(self, data, labels):
        self.model.fit(data, labels, shuffle=True, epochs=20, batch_size=4)

    def test(self, data, labels):
        return self.model.evaluate(data, labels, batch_size=4)
