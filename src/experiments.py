from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_cleaner import load_data, get_preprocessor
from configs.config_loader import SEED, TEST_SIZE


def split():
    x, y = load_data()

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y,
        random_state=SEED,
        test_size=TEST_SIZE
    )

    preprocessor = get_preprocessor()

    return xtrain, xtest, ytrain, ytest