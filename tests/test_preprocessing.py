import sys
sys.path.insert(0, "src")

import pytest
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

def test_load_datafile_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_data('data/processed/nonexistent_file.csv')

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

def test_build_preprocessing_pipeline_empty_numerical_raises() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        build_preprocessing_pipeline(
            numerical_features=[],
            nominal_features=['some_nominal_col'],
            ordinal_features=['some_ordinal_col'],
        )

def test_build_preprocessing_pipeline_transforms_shape() -> None:
    df = load_data(DATA_PATH)
    numerical_features = ['Age', 'Monthly_Income']
    nominal_features = ['Gender', 'Department']
    ordinal_features = ['Work_Life_Balance']

    pipeline = build_preprocessing_pipeline(numerical_features, nominal_features, ordinal_features)
    transformed = pipeline.fit_transform(df)

    assert transformed.shape[0] == len(df)
    assert not np.isnan(transformed).any()
    assert transformed.shape[1] >= len(numerical_features) + len(ordinal_features) + len(nominal_features)
