import yaml
import mlflow
import mlflow.sklearn

from train_model import train_model

def load_config(filepath):
    with open(filepath, "r") as config:
        return yaml.safe_load(config)


def main() -> None:
    CONFIG_PATH = "configs/config.yaml"
    config = load_config(CONFIG_PATH)

    mlflow.set_experiment(config['project'].get('name'))

    model_types = config['model_types']
    for model_type in model_types:
        print()
        print('model_types', config.get(model_type))
        train_model(config, model_type)


if __name__ == "__main__":
    main()