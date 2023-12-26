from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, F1Score

from util.config import Config
from util.data import Dataset


def get_model(config: Config, data: Dataset):
    is_multiclass = len(data.y_train.shape) > 1
    loss = 'categorical_crossentropy' if is_multiclass else 'binary_crossentropy'

    model = Sequential()

    for _ in range(config.depth):
        model.add(Dense(config.width, activation='relu'))
    
    if is_multiclass:
        model.add(Dense(data.y_train.shape[1], activation='softmax'))
    else:
        model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=config.learning_rate_init),
        metrics=['accuracy', AUC(curve='ROC'), F1Score()]
    )

    return model