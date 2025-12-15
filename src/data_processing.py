
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def prepare_model_dataset(df: pd.DataFrame):
    """
    Transforms raw transaction data into a model-ready dataset
    and returns the processed dataframe and preprocessing pipeline.
    """

    df = df.copy()

    # -----------------------------
    # Temporal Feature Extraction
    # -----------------------------
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    # -----------------------------
    # Aggregate Customer Features
    # -----------------------------
    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_transaction_amount=("Amount", "std"),
        )
        .reset_index()
    )

    df = df.merge(agg_df, on="CustomerId", how="left")

    # -----------------------------
    # Feature Selection
    # -----------------------------
    numeric_features = [
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
    ]

    categorical_features = [
        "ProductCategory",
        "ChannelId",
        "CountryCode",
        "PricingStrategy",
    ]

    # -----------------------------
    # Pipelines
    # -----------------------------
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return df, preprocessor

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
