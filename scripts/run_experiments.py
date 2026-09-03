import sys
sys.path.insert(0, "src")

import yaml
import mlflow

from train_model import train_model

def load_config(filepath):
    with open(filepath, "r") as config:
        return yaml.safe_load(config)


def main() -> None:
    DEFAULT_CONFIG_PATH = "configs/config.yml"
    config = load_config(DEFAULT_CONFIG_PATH)

    mlflow.set_experiment(config['project'].get('name'))

    model_types = config['model_types']
    for model_type in model_types:
        print()
        print('model_types', config.get(model_type))
        train_model(config, model_type)


if __name__ == "__main__":
    main()