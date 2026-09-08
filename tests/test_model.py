import sys
sys.path.insert(0, "src")

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.build_model import build_model
from src.evaluate_model import calculate_metrics
from src.preprocess_data import (
    load_data,
    drop_features,
    split_dataframe,
    split_features_target,
    build_preprocessing_pipeline,
)

TARGET = 'Attrition'
DATA_PATH = 'data/processed/employee_attrition_dataset.csv'


def test_model_predictions_have_correct_type_and_shape() -> None:
    df = load_data(DATA_PATH)
    df = drop_features(df, ['Employee_ID'])
    features, target = split_features_target(df, TARGET)
    features_train, features_val, target_train, _ = split_dataframe(features, target, 0.2)

    target_train = target_train.map({'No': 0, 'Yes': 1})

    preprocessor = build_preprocessing_pipeline(
        numerical_features=['Age', 'Monthly_Income'],
        nominal_features=['Gender', 'Department'],
        ordinal_features=['Job_Level'],
        ordinal_order=[1, 2, 3, 4, 5],
    )
    model_config = {'name': 'LogisticRegression', 
                    'params': {'max_iter': 100, 'random_state': 1234}}
    model_pipeline = build_model(preprocessor, model_config)

    model_pipeline.fit(features_train, target_train)
    predictions = model_pipeline.predict(features_val)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(features_val),)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_model_meets_minimum_performance_threshold() -> None:
    df = load_data(DATA_PATH)
    df = drop_features(df, ['Employee_ID'])
    features, target = split_features_target(df, TARGET)
    features_train, features_val, target_train, target_val = split_dataframe(features, target, 0.2)

    target_train = target_train.map({'No': 0, 'Yes': 1})
    target_val = target_val.map({'No': 0, 'Yes': 1})

    preprocessor = build_preprocessing_pipeline(
        numerical_features=['Age', 'Monthly_Income'],
        nominal_features=['Gender', 'Department'],
        ordinal_features=['Job_Level'],
        ordinal_order=[1, 2, 3, 4, 5],
    )
    model_config = {
        'name': 'LogisticRegression',
        'params': {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 1234},
    }
    model_pipeline = build_model(preprocessor, model_config)
    model_pipeline.fit(features_train, target_train)
    predictions = model_pipeline.predict(features_val)
    metrics = calculate_metrics(target_val, predictions)

    dummy = DummyClassifier(strategy='most_frequent', random_state=1234)
    dummy.fit(features_train, target_train)
    dummy_predictions = dummy.predict(features_val)
    dummy_metrics = calculate_metrics(target_val, dummy_predictions)

    # assert metrics['Accuracy Score'] >= dummy_metrics['Accuracy Score']
    assert metrics['f1 Score'] > 0.0
