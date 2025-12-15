import pandas as pd


def parse_transaction_time(df, time_col="TransactionStartTime"):
    """Parse transaction timestamp and extract temporal features."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    df["transaction_hour"] = df[time_col].dt.hour
    df["transaction_day"] = df[time_col].dt.day
    df["transaction_month"] = df[time_col].dt.month
    df["transaction_year"] = df[time_col].dt.year

    return df
