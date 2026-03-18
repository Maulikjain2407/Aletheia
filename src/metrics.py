import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import cross_validate


def model_eval(model, xtrain, ytrain, xtest, ytest, cv):

    cv_results = cross_validate(
        model,
        xtrain,
        ytrain,
        cv=cv,
        scoring=["accuracy", "f1", "neg_log_loss"],
        return_train_score=True
    )

    def summarize(arr):
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    train_acc_mean, train_acc_sd = summarize(cv_results["train_accuracy"])
    val_acc_mean, val_acc_sd = summarize(cv_results["test_accuracy"])

    train_f1_mean, train_f1_sd = summarize(cv_results["train_f1"])
    val_f1_mean, val_f1_sd = summarize(cv_results["test_f1"])

    train_log_mean, train_log_sd = summarize(-cv_results["train_neg_log_loss"])
    val_log_mean, val_log_sd = summarize(-cv_results["test_neg_log_loss"])

    generalisation_gap = train_acc_mean - val_acc_mean

    model.fit(xtrain, ytrain)
    y_test_pred = model.predict(xtest)
    y_test_prob = model.predict_proba(xtest)[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(ytest, y_test_pred),
        "f1": f1_score(ytest, y_test_pred),
        "logloss": log_loss(ytest, y_test_prob)
    }

    evaluations = {
        "accuracy": {
            "training": {"mean": train_acc_mean, "sd": train_acc_sd},
            "validation": {"mean": val_acc_mean, "sd": val_acc_sd}
        },
        "f1": {
            "training": {"mean": train_f1_mean, "sd": train_f1_sd},
            "validation": {"mean": val_f1_mean, "sd": val_f1_sd}
        },
        "logloss": {
            "training": {"mean": train_log_mean, "sd": train_log_sd},
            "validation": {"mean": val_log_mean, "sd": val_log_sd}
        }
    }

    return {
        "evaluations": evaluations,
        "generalisation_gap": float(generalisation_gap),
        "test_metrics": test_metrics
    }