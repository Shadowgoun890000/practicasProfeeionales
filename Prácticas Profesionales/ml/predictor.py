import pandas as pd

from ml.schema import MODEL_FEATURES


def get_available_features(df: pd.DataFrame) -> list[str]:
    return [col for col in MODEL_FEATURES if col in df.columns]


def build_model_input(df: pd.DataFrame) -> pd.DataFrame:
    features = get_available_features(df)
    return df[features].copy()


def predict_dataframe(model, df: pd.DataFrame):
    if model is None:
        return None

    x = build_model_input(df)

    if x.empty:
        return None

    return model.predict(x)