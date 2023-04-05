Spaceship Titanic
==============================

## Description
Репозиторий демонстрирует реализацию проекта "MLOps на примере задачи Spaceship Titanic"

Исходные данные:
https://www.kaggle.com/competitions/spaceship-titanic


## Основные шаги:
1. Реализован baseline в рамках соревнования на Kaggle
2. Создан локальный репозиторий с структурой проекта cookiecutter data science template
3. Создана виртуальная среда (env) и установлены необходимые для проекта зависимости
4. Добавлены стилистический линтер (flake8) и авто-форматеры (black+isort)
5. Настроен .gitignore
6. Создан удаленный репозиторий (GitLab), синхронизирован с локальным по SSH
7. Удаленный репозиторий настроен под github-flow
8. Исходный код организован по модулям
9. CLI (click)
10. Создан CI-пайплайн (black+flake8)
11. Реализованы тесты (pytest, great_expectations) и добавлены в CI-пайплайн
12. Установлен и сконфигурирован DVC
13. Настроен пайплайн запуска модулей с помощью DVC (dvc.yaml, dvc repro)
14. Реализован подбор гиперпараметров моделей с помощью Optuna для моделей xgboost, catboost и lightgbm
15. Интегрированы Optuna и MLFlow для трекинга экспериментов
16. Настроены в MLflow логирование параметров, метрик и моделей (по сценарию 3)
17. Развернут MLflow по сценарию 4: СУБД (PostgreSQL) + хранилище артифактов (MinIO) в Docker-контейнерах
18. Проведены эксперименты с логированием модели
19. Реализован получение Staging-модели из MLFlow models и прогноз на тестовых данных
20. Модель запущена как сервис в Docker-контейнере

## Tехнологический стек проекта:
- Python 3.10
- Зависимости: `pip` + `env` / `poetry`
- CLI: `click`
- Шаблон проекта: `cookiecutter`
- Версионирование кода: `GitLab` / `GitHub`
- Версионирование данных: `dvc` + `minio`
- Workflow менеджер: `dvc`
- Контроль codestyle:
	- Линтеры: `pylint`, `flake8`
	- Форматтеры: `black`, `isort`
- Трекинг экспериментов: `mlflow` (Scenario 4: DB PostgreSQL + S3 minio)
- ML: `xgboost`, `catboost`, `lightgbm`
- Оптимизация гиперпараметров: `optuna`
- Тестирование: `pytest`, `great_expectations`
- API: `FastAPI` + `Uvicorn`
- CI/CD: `gitlab CI` + `Docker` + `nexus`



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
