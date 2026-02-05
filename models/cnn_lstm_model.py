from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Dropout,
    Flatten, RepeatVector, LSTM, Dense
)

def build_model(input_shape, num_classes=3):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(RepeatVector(4))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax"))

    return model
