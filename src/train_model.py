# import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn

from sklearn.dummy import DummyClassifier

from preprocess_data import (
    load_data, 
    drop_features, 
    split_dataframe, 
    split_features_target, 
    build_preprocessing_pipeline
)
from build_model import build_model
from evaluate_model import calculate_metrics, print_metrics



def train_model(config: dict[str, any], model_type: str):
    data_config = config['data']

    filepath = data_config.get('filepath')
    df = load_data(filepath)

    df = drop_features(df, data_config.get('features_to_drop'))
    features, target = split_features_target(df, data_config.get('target'))
    features_train, features_val, target_train, target_val = split_dataframe(features, target, config['training'].get('test_size'))

    feature_types = data_config.get('feature_types')
    preprocessor_pipeline = build_preprocessing_pipeline(
        numerical_features= feature_types.get('numerical'),
        nominal_features= feature_types.get('nominal'),
        ordinal_features= feature_types.get('ordinal'),
        ordinal_order=[1,2,3,4,5]
    )

    model_config = config.get(model_type)
    model_pipeline = build_model(preprocessor_pipeline, model_config)

    target_train_processed = target_train.map({'No': 0, 'Yes': 1}) 
    target_val_processed = target_val.map({'No': 0, 'Yes': 1})

    with mlflow.start_run(run_name=model_type) as active_run:
        mlflow.log_params(model_config.get('params'))  

        model_pipeline.fit(features_train, target_train_processed)
   
        print(model_pipeline.named_steps['model'].get_params())
        predictions = model_pipeline.predict(features_val)

        # print('PREDICC UNIQUE')
        # print(np.unique(predictions, return_counts=True))
        # print('AFTER MODEL PREDICT')
   
        metrics_dict = calculate_metrics(target_val_processed, predictions)
    
        mlflow.log_metrics(metrics_dict)
        mlflow.sklearn.log_model(
            model_pipeline, 
            "model",
            input_example=features_train.iloc[:5]
        )

        run_id = mlflow.active_run().info.run_id

    # --------------------------
    # probs = model_pipeline.predict_proba(features_val)[:, 1]
    # print(np.sort(probs)[-20:])   # the 20 highest predicted "Yes" probabilities
    # print(probs.mean())            # average predicted probability across all validation rows
    # print(probs.max()) 

    # -------------- DUMMY CLASSIFIER
    # dummy = DummyClassifier(strategy="most_frequent")
    # dummy.fit(features_train, target_train_processed)
    # print("DUMMY")
    # print(accuracy_score(target_val_processed, dummy.predict(features_val)))


# def main() -> None:
#     CONFIG_PATH = "configs/config.yaml"
#     config = load_config(CONFIG_PATH)

#     train_model(config)

# if __name__ == "__main__":
#     main()