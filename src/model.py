from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def build_model():

    model = Sequential([

        # Input
        Input(shape=(128, 128, 3)),

        # Convolution Block 1
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        # Convolution Block 2
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        # Convolution Block 3
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        # Classification
        Flatten(),

        Dense(128, activation="relu"),

        Dropout(0.5),

        # 36 ISL classes
        Dense(36, activation="softmax")
    ])

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()