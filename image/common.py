import gc

from image.src.config import Config
from image.src.util import BATCH_SIZE, get_data, get_model


def evaluate(config: Config, dataset: str = "mnist", *args, **kwargs) -> float:
    data = get_data(dataset)
    model = get_model(data, config, 10, dataset=dataset)

    try:
        model.fit(data.x_train, data.y_train, batch_size=BATCH_SIZE, epochs=50)
        scores = model.evaluate(data.x_test, data.y_test, verbose=0)[-1]
        _ = gc.collect()
    except ValueError:
        return 100.0

    return scores
