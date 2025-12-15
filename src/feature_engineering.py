import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.utils import parse_transaction_time



def create_customer_aggregates(df):
    """
    Create customer-level aggregate features.
    """
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

    return agg_df


def build_feature_pipeline(numerical_features, categorical_features):
    """
    Build sklearn pipeline for preprocessing.
    """

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
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def prepare_model_dataset(raw_df):
    """
    Full feature engineering pipeline:
    - temporal features
    - aggregate features
    - preprocessing pipeline
    """

    # 1. Parse datetime & extract temporal features
    df = parse_transaction_time(raw_df)

    # 2. Create aggregates
    agg_df = create_customer_aggregates(df)

    # 3. Merge aggregates back
    df = df.merge(agg_df, on="CustomerId", how="left")

    # 4. Feature lists
    numerical_features = [
        "Amount",
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
    ]

    categorical_features = [
        "ProductCategory",
        "ChannelId",
        "CountryCode",
        "PricingStrategy",
    ]

    # 5. Build pipeline
    preprocessor = build_feature_pipeline(
        numerical_features, categorical_features
    )

    return df, preprocessor
