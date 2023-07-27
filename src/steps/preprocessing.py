from pathlib import Path
import os
import pandas as pd
import numpy as np
from pickle import dump
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# constants
LOCAL = os.getenv("LOCAL", False)

if LOCAL:
    BASE_FILEPATH = "/opt/ml/processing"
    DATA_FILEPATH = "/opt/ml/processing/input/data.csv"
else:
    BASE_FILEPATH = "/data"
    DATA_FILEPATH = "/data/data.csv"


# what kind of data do we have with train, validation and test?
def save_splits(base_dir: str, train, validation, test) -> None:
    train_path = Path(base_dir) / "train"
    validation_path = Path(base_dir) / "validation"
    test_path = Path(base_dir) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", index=False, header=False)
    pd.DataFrame(validation).to_csv(
        validation_path / "validation.csv", index=False, header=False
    )
    pd.DataFrame(test).to_csv(test_path / "test.csv", index=False, header=False)


def save_sk_pipeline(base_dir: str, pipeline) -> None:
    pipeline_path = Path(base_dir) / "pipeline"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    dump(pipeline, open(pipeline_path / "pipeline.pkl", "wb"))


def save_classes(base_dir: str, classes) -> None:
    path = Path(base_dir) / "classes"
    path.mkdir(parents=True, exist_ok=True)

    np.asarray(classes).tofile(path / "classes.csv", sep=",")


def save_baseline(base_dir: str, df_train: pd.DataFrame, df_test: pd.DataFrame):
    for split, data in [("train", df_train), ("test", df_test)]:
        baseline_path = Path(base_dir) / f"{split}-baseline"
        baseline_path.mkdir(parents=True, exist_ok=True)

        df = data.copy()
        df.to_json(
            baseline_path / f"{split}-baseline.json", orient="records", lines=True
        )


def preprocess(base_dir: str, data_filepath: str):
    df = pd.read_csv(data_filepath)

    numeric_features = df.select_dtypes(include=["float64"]).columns.tolist()
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, ["island"]),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessing", preprocessor)])

    df.drop(["sex"], axis=1, inplace=True)
    df = df.sample(frac=1, random_state=42)

    df_train, temp = train_test_split(df, test_size=0.3)
    df_validation, df_test = train_test_split(temp, test_size=0.5)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train.species)
    y_validation = label_encoder.transform(df_validation.species)
    y_test = label_encoder.transform(df_test.species)

    save_baseline(base_dir, df_train, df_test)

    df_train = df_train.drop(["species"], axis=1)
    df_validation = df_validation.drop(["species"], axis=1)
    df_test = df_test.drop(["species"], axis=1)

    X_train = pipeline.fit_transform(df_train)
    X_validation = pipeline.transform(df_validation)
    X_test = pipeline.transform(df_test)

    train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
    validation = np.concatenate(
        (X_validation, np.expand_dims(y_validation, axis=1)), axis=1
    )
    test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)

    save_splits(base_dir, train, validation, test)
    save_sk_pipeline(base_dir, pipeline=pipeline)
    save_classes(base_dir, label_encoder.classes_)


if __name__ == "__main__":
    preprocess(base_dir=BASE_FILEPATH, data_filepath=DATA_FILEPATH)
