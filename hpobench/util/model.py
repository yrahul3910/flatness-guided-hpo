from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from util.config import Config
from util.data import Dataset


def get_model(config: Config, data: Dataset):
    model = Sequential()

    for _ in range(config.depth):
        model.add(Dense(config.width, activation='relu'))
    model.add(Dense(data.y_train.shape[1], activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=config.learning_rate_init),
        metrics=['accuracy', AUC(curve='ROC')]
    )

    return model