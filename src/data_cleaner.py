import pandas as pd

from configs.config_loader import training_path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def get_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputerN", SimpleImputer(strategy="mean"))
    ])

    categorical_pipeline = Pipeline([
        ("imputerC", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, ["Age", "Fare", "SibSp", "Parch"]),
        ("categorical", categorical_pipeline, ["Sex", "Embarked"])
    ])

    return preprocessing


def load_data():
    df = pd.read_csv(training_path)
    df = df.drop(["PassengerId", "Cabin", "Name", "Ticket"], axis=1)

    x = df.drop("Survived", axis=1)
    y = df["Survived"]

    return x, y