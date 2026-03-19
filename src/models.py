from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from src.data_cleaner import get_preprocessor

from configs.config_loader import model_parameters as mp


def get_model():

    preprocessor = get_preprocessor()

    models = {

        "logistic": {
            "model": Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000))
            ]),
            "parameters": {
                "classifier__C": mp.logistic_parameters["C"]
            }
        },

        "decision_trees": {
            "model": Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", DecisionTreeClassifier())
            ]),
            "parameters": {
                "classifier__criterion": mp.decision_tree_parameters["criterion"],
                "classifier__max_depth": mp.decision_tree_parameters["max_depth"],
                "classifier__min_samples_leaf": mp.decision_tree_parameters["min_samples_leaf"]
            }
        },

        "random_forest": {
            "model": Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", RandomForestClassifier())
            ]),
            "parameters": {
                "classifier__n_estimators": mp.random_forest_parameters["n_estimators"],
                "classifier__max_depth": mp.random_forest_parameters["max_depth"],
                "classifier__criterion": mp.random_forest_parameters["criterion"]
            }
        },

        "XG_boost": {
            "model": Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", XGBClassifier())
            ]),
            "parameters": {
                "classifier__n_estimators": mp.xgboost_parameters["n_estimators"],
                "classifier__max_depth": mp.xgboost_parameters["max_depth"],
                "classifier__learning_rate": mp.xgboost_parameters["learning_rate"]
            }
        }

    }

    return models