# -*- coding: utf-8 -*-
import catboost
import click
import joblib as jb
import lightgbm
import mlflow
import optuna
import pandas as pd
import xgboost
from optuna.integration import CatBoostPruningCallback
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("optuna-integration-new")

global train_x, valid_x, train_y, valid_y


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def optuna_optimization(input_filepath: str, output_filepath: str):
    global train_x, valid_x, train_y, valid_y

    train_preproc = pd.read_csv(input_filepath)
    features_train = train_preproc.drop(["Transported"], axis=1)
    target_train = train_preproc.Transported.astype(int)
    train_x, valid_x, train_y, valid_y = train_test_split(
        features_train, target_train, test_size=0.25
    )

    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        create_experiment=False,
        metric_name="Accuracy",
    )

    study = optuna.create_study(
        pruner=optuna.pruners.HyperbandPruner(),
        direction="maximize",
    )
    study.optimize(objective, n_trials=10, callbacks=[mlflc])

    mlflow.start_run()

    best_params = study.best_params
    model = catboost.CatBoostClassifier(**best_params, verbose=False)
    model.fit(features_train, target_train)
    jb.dump(model, output_filepath)

    mlflow.log_params(best_params)
    mlflow.log_metric("Accuracy", model.score(valid_x, valid_y))

    signature = infer_signature(valid_x, model.predict(valid_x))
    mlflow.catboost.log_model(
        cb_model=model,
        artifact_path="model.clf",
        registered_model_name="model_catboost",
        signature=signature,
    )
    mlflow.end_run()

    df = mlflow.search_runs(filter_string="metrics.Accuracy > 0.8")
    run_id = df.loc[df["metrics.Accuracy"].idxmin()]["run_id"]
    print("df", df)
    print("run_id", run_id)


def objective(trial: optuna.Trial) -> float:
    param = {
        "objective": "Logloss",
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.01, 0.1, log=True
        ),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["Ordered", "Plain"]
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
        "eval_metric": "Accuracy",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    gbm = catboost.CatBoostClassifier(**param)

    pruning_callback = CatBoostPruningCallback(trial, "Accuracy")
    gbm.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=0,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    accuracy = gbm.score(valid_x, valid_y)

    return accuracy


if __name__ == "__main__":
    optuna_optimization()
