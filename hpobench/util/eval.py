from typing import List
import gc

from util.config import Config
from util.data import Dataset
from util.model import get_model

from keras.callbacks import EarlyStopping


def eval(config: Config, data: Dataset, *args, **kwargs) -> List[float]:
    if isinstance(config, dict):
        config = Config(**config)

    model = get_model(config, data)

    try:
        early_stop = EarlyStopping(monitor='loss', patience=5, min_delta=1e-3)
        model.fit(data.x_train, data.y_train, batch_size=config.batch_size, epochs=200, callbacks=[early_stop])
        score = model.evaluate(data.x_test, data.y_test, verbose=0)
        
        print(f'Score: {score}')

        gc.collect()
    except ValueError:
        return [0., 0., 0.]

    return score
