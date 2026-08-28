from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    HistGradientBoostingClassifier
)


model_dict = {
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier': RandomForestClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier
}

def build_model(preprocessor: ColumnTransformer, model_config: dict[str: any]) -> Pipeline:
    model = model_dict[model_config.get('name')]
    params = model_config.get('params')

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model(**params))
    ])

  
