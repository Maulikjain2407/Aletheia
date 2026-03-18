from sklearn.model_selection import GridSearchCV
from configs.config_loader import N_JOBS, SCORING


def tuner(model, parameters, xtrain, ytrain, cv):

    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=cv,
        n_jobs=N_JOBS,
        scoring=SCORING,
    )

    search.fit(xtrain, ytrain)

    return search.best_estimator_, search.best_params_
