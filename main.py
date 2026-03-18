import time
from sklearn.model_selection import StratifiedKFold

from src.experiments import split
from src.models import get_model
from src.tuning import tuner
from src.metrics import model_eval

from configs.config_loader import CV, SEED, TEST_SIZE, DATASET_NAME, SCORING, N_JOBS
from logger.experiment_logger import experiment_logger

start = time.time()


def main():
    x_train, x_test, y_train, y_test = split()

    cv_strategy = StratifiedKFold(
        n_splits=CV,
        shuffle=True,
        random_state=SEED
    )

    models = get_model()
    trained_models = {}

    for name, model_info in models.items():

        model_candidate, params_candidate = tuner(
            model=model_info["model"],
            parameters=model_info["parameters"],
            xtrain=x_train,
            ytrain=y_train,
            cv=cv_strategy
        )

        results = model_eval(
            model=model_candidate,
            xtrain=x_train,
            ytrain=y_train,
            xtest=x_test,
            ytest=y_test,
            cv=cv_strategy
        )

        trained_models[name] = {
            "model": model_candidate,
            "best_params": params_candidate,
            "evaluations": results["evaluations"],
            "generalisation_gap": results["generalisation_gap"],
            "test_metrics": results["test_metrics"]
        }

        print(f"\n{name}")
        print("Best Params:", params_candidate)

        for metric in results["evaluations"]:
            t = results["evaluations"][metric]["training"]
            v = results["evaluations"][metric]["validation"]

            print(f"{metric.upper()}:")
            print(f"  Train: {t['mean']:.3f} ± {t['sd']:.3f}")
            print(f"  Validation: {v['mean']:.3f} ± {v['sd']:.3f}")

        gap = results["generalisation_gap"]
        print(f"Generalisation Gap: {gap:.3f}")

    return trained_models


if __name__ == "__main__":
    trained_models = main()

    logger = experiment_logger()
    logger.json_writer(
        trained_models=trained_models,
        dataset_name=DATASET_NAME,
        test_size=TEST_SIZE,
        seed=SEED,
        scoring=SCORING,
        n_jobs=N_JOBS,
        cv=CV
    )

end = time.time()
print("TOTAL TIME =", end - start)