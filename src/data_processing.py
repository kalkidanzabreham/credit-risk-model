import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw transaction data from CSV.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {filepath}: {e}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using explicit, justified strategies.
    - Numerical features: median imputation (robust to skewness)
    - Categorical features: mode imputation
    """
    df = df.copy()

    numerical_cols = ["Amount", "Value"]
    categorical_cols = ["ProductCategory", "ChannelId", "CountryCode", "PricingStrategy"]

    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df
