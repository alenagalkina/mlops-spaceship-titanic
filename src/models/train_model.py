# -*- coding: utf-8 -*-
import os

import catboost as cb
import click
import joblib as jb
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature

# from optuna.integration import (
#     CatBoostPruningCallback,
#     LightGBMPruningCallback,
#     XGBoostPruningCallback,
# )
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

load_dotenv()
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(remote_server_uri)
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

global train_x, valid_x, train_y, valid_y


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def train_model(input_filepath: str, output_filepath: str):
    global train_x, valid_x, train_y, valid_y

    train_preproc = pd.read_csv(input_filepath)
    features_train = train_preproc.drop(["Transported"], axis=1)
    target_train = train_preproc.Transported.astype(int)
    train_x, valid_x, train_y, valid_y = train_test_split(
        features_train,
        target_train,
        test_size=0.25,
    )

    signature = infer_signature(train_x, train_y)

    mlflow.set_experiment("optuna-cb")
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        create_experiment=False,
        metric_name="ROC_AUC",
    )

    study = optuna.create_study(
        pruner=optuna.pruners.HyperbandPruner(),
        direction="maximize",
    )
    study.optimize(objective_cb, n_trials=10, callbacks=[mlflc])

    catboost_best_params = study.best_params
    catboost_best_value = study.best_value
    catboost_best_trial = study.best_trial

    with mlflow.start_run():
        mlflow.log_params(catboost_best_params)

        model = cb.CatBoostClassifier(**catboost_best_params, silent=True)
        model.fit(train_x, train_y)

        jb.dump(model, output_filepath)

        mlflow.log_metric(
            "ROC_AUC",
            roc_auc_score(valid_y, np.rint(model.predict(valid_x))),
        )

        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            registered_model_name="catboost_model",
            signature=signature,
        )

    mlflow.set_experiment("optuna-xgb")
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        create_experiment=False,
        metric_name="ROC_AUC",
    )

    study = optuna.create_study(
        pruner=optuna.pruners.HyperbandPruner(),
        direction="maximize",
    )
    study.optimize(
        objective_xgb,
        n_trials=10,
        callbacks=[mlflc],
    )
    xgboost_best_params = study.best_params
    xgboost_best_value = study.best_value
    xgboost_best_trial = study.best_trial

    with mlflow.start_run():
        mlflow.log_params(xgboost_best_params)

        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)
        model = xgb.train(xgboost_best_params, dtrain)

        jb.dump(model, output_filepath)

        mlflow.log_metric(
            "ROC_AUC",
            roc_auc_score(valid_y, np.rint(model.predict(dvalid))),
        )

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="xgboost_model",
            signature=signature,
        )

    mlflow.set_experiment("optuna-lgb")
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        create_experiment=False,
        metric_name="ROC_AUC",
    )

    study = optuna.create_study(
        pruner=optuna.pruners.HyperbandPruner(),
        direction="maximize",
    )
    study.optimize(
        objective_lgb,
        n_trials=10,
        callbacks=[mlflc],
    )

    lgboost_best_params = study.best_params
    lgboost_best_value = study.best_value
    lgboost_best_trial = study.best_trial

    with mlflow.start_run():
        mlflow.log_params(lgboost_best_params)

        dtrain = lgb.Dataset(train_x, label=train_y)

        model = lgb.train(lgboost_best_params, dtrain, verbose_eval=False)

        jb.dump(model, output_filepath)

        mlflow.log_metric(
            "ROC_AUC",
            roc_auc_score(valid_y, np.rint(model.predict(valid_x))),
        )

        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name="lgboost_model",
            signature=signature,
        )

    print("catboost_best_params", catboost_best_params)
    print("catboost_best_value", catboost_best_value)
    print("catboost_best_trial", catboost_best_trial.number)
    print("xgboost_best_params", xgboost_best_params)
    print("xgboost_best_value", xgboost_best_value)
    print("xgboost_best_trial", xgboost_best_trial.number)
    print("lgboost_best_params", lgboost_best_params)
    print("lgboost_best_value", lgboost_best_value)
    print("lgboost_best_trial", lgboost_best_trial.number)


def objective_cb(trial: optuna.Trial) -> float:

    param = {
        "objective": "Logloss",
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel",
            0.01,
            0.1,
            log=True,
        ),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical(
            "boosting_type",
            ["Ordered", "Plain"],
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type",
            ["Bayesian", "Bernoulli", "MVS"],
        ),
        "used_ram_limit": "3gb",
        "eval_metric": "AUC",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    gbm = cb.CatBoostClassifier(**param)

    # pruning_callback = CatBoostPruningCallback(trial, "Accuracy")
    gbm.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=0,
        early_stopping_rounds=100,
        # callbacks=[pruning_callback],
    )

    # pruning_callback.check_pruned()

    print("roc_auc_score", roc_auc_score(valid_y, np.rint(gbm.predict(valid_x))))
    return roc_auc_score(valid_y, np.rint(gbm.predict(valid_x)))


def objective_xgb(trial: optuna.Trial) -> float:

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "eval_metric": "auc",
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy",
            ["depthwise", "lossguide"],
        )

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type",
            ["uniform", "weighted"],
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type",
            ["tree", "forest"],
        )
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # pruning_callback = XGBoostPruningCallback(trial, "validation-auc")

    bst = xgb.train(
        param,
        dtrain,
        evals=[(dvalid, "validation")],
        # callbacks=[pruning_callback],
    )
    print("roc_auc_score", roc_auc_score(valid_y, np.rint(bst.predict(dvalid))))
    return roc_auc_score(valid_y, np.rint(bst.predict(dvalid)))


def objective_lgb(trial: optuna.Trial) -> float:

    dtrain = lgb.Dataset(train_x, label=train_y)
    dvalid = lgb.Dataset(valid_x, label=valid_y)

    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    # pruning_callback = LightGBMPruningCallback(trial, "auc")
    gbm = lgb.train(
        param,
        dtrain,
        valid_sets=[dvalid],
        # callbacks=[pruning_callback],
        verbose_eval=False,
    )
    print("roc_auc_score", roc_auc_score(valid_y, np.rint(gbm.predict(valid_x))))
    return roc_auc_score(valid_y, np.rint(gbm.predict(valid_x)))


if __name__ == "__main__":
    train_model()
