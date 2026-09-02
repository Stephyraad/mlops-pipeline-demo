import sys
sys.path.insert(0, "src")

import pandas as pd
import mlflow

def get_runs_by_experiment(experiment_name: str) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    df_runs = mlflow.search_runs(experiment_ids=experiment.experiment_id)

    return df_runs

def get_best_run(experiment_name: str, metric: str, is_ascending: bool = False):
    df_runs = get_runs_by_experiment(experiment_name)
    sorted_runs = df_runs.sort_values(f"metrics.{metric}", ascending=is_ascending)

    best_run = sorted_runs.iloc[0]
    print(best_run)

    return best_run


def main() -> None:
    DEFAULT_EXPERIMENT_NAME = 'employee_attrition_prediction'

    # MLflow metrics options  are: f1 Score, Recall Score, Accuracy Score, Precision Score 
    DEFAULT_METRIC = 'f1 Score'
    get_best_run(DEFAULT_EXPERIMENT_NAME, DEFAULT_METRIC, False)
if __name__ == "__main__":
    main()