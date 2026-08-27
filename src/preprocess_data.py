

import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split


DEFAULT_TARGET = 'Attrition'

def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found"
        )
    return pd.read_csv(path)


def print_dataframe_info(df: pd.DataFrame) -> None:
    print(df.info())
    print(df.head())


def drop_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    return df.drop(columns=columns)


def split_features_target(df: pd.DataFrame, target_col: str = DEFAULT_TARGET) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    features = drop_features(df, [target_col])
    target = df[target_col]
    return features, target


def split_dataframe(
        features: pd.DataFrame,target: pd.Series, split_size = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    features_train, features_test, target_train, target_test = train_test_split(
      features, 
      target,
      test_size=split_size,
      random_state=1234,
    )
    return features_train, features_test, target_train, target_test


def build_features_preprocessor(
        numerical_features: list[str],
        nominal_features: list[str],
        ordinal_features: list[str],
        ordinal_order: list[str] = None,
    ) -> ColumnTransformer:
    if not numerical_features:
        raise ValueError('numeric_colums cannot be empty')

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    nominal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ordinal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[ordinal_order] if ordinal_order else 'auto', handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    transformers = [
        ('numerical', numerical_pipeline, numerical_features),
        ('nominal', nominal_pipeline, nominal_features),
        ('ordinal', ordinal_pipeline, ordinal_features)
    ]

    return ColumnTransformer(transformers=transformers, remainder='drop')
