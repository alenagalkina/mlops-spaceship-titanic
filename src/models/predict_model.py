# -*- coding: utf-8 -*-
import mlflow
import numpy as np
import pandas as pd
import catboost
from sklearn.metrics import roc_auc_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")


def staging_model_prediction():
    # Load model
    model_name = "catboost_model"
    stage = "Staging"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

    # Inference
    data = pd.read_csv("data/interim/train_postproc.csv")
    test = data.drop(["Transported"], axis=1)
    print(roc_auc_score(data.Transported.astype(int), np.rint(model.predict(test))))


def get_run_max_metric():
    df = mlflow.search_runs(
        experiment_ids=["1", "2", "3"], filter_string="metrics.ROC_AUC > 0.8"
    )
    run_id = df.loc[df["metrics.ROC_AUC"].idxmin()]["run_id"]
    print("run_id", run_id)
    run_data = mlflow.get_run(run_id=run_id)
    print("params: {}".format(run_data.data.params))
    print("metrics: {}".format(run_data.data.metrics))


# def max_metric_model_prediction():
#     df = mlflow.search_runs(experiment_ids=["1"], filter_string="metrics.ROC_AUC > 0.8")
#     run_id = df.loc[df["metrics.ROC_AUC"].idxmin()]["run_id"]
#     run_data = mlflow.get_run(run_id=run_id)

#     print(run_data.data.params["depth"])
#     data = pd.read_csv("data/interim/train_postproc.csv")
#     model = catboost.CatBoostClassifier(**run_data.data.params)
#     model.fit(data.drop(["Transported"], axis=1), data.Transported.astype(int))
#     # Inference

#     test = data.drop(["Transported"], axis=1)
#     print(roc_auc_score(data.Transported.astype(int), np.rint(model.predict(test))))


if __name__ == "__main__":
    staging_model_prediction()
    get_run_max_metric()
    # max_metric_model_prediction()
