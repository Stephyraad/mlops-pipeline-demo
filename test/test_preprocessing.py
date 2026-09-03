import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from src.preprocess_data import (
    load_data,
    drop_features,
    split_dataframe,
    split_features_target,
    build_preprocessing_pipeline,
)


DATA_PATH = 'data/processed/employee_attrition_dataset.csv'

def test_drop_features() -> None:
    features_to_drop = ['Employee_ID']
    df = load_data(DATA_PATH)

    assert ("Employee_ID" not in drop_features(df, features_to_drop).columns)


def test_split_features_target() -> None:
    TARGET = 'Attrition'
    df = load_data(DATA_PATH)
    features, target = split_features_target(df, TARGET)

    assert TARGET not in features.columns
    assert target.name == TARGET
    assert len(features) == len(target) == len(df)

def test_split_dataframe() -> None:
    TARGET = 'Attrition'
    SPLIT_SIZE = 0.2
    df = load_data(DATA_PATH)
    features, target = split_features_target(df, TARGET)

    features_train, features_test, target_train, target_test = split_dataframe(
        features, target, SPLIT_SIZE
    )

    assert len(features_train) == len(target_train)
    assert len(features_test) == len(target_test)
    assert len(features_train) + len(features_test) == len(features)
    assert len(features_test) == round(len(features) * SPLIT_SIZE)

