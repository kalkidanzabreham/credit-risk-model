from src.target_engineering import calculate_rfm
import pandas as pd


def test_rfm_columns_exist():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionStartTime": pd.to_datetime(["2023-01-01", "2023-01-05", "2023-01-03"]),
        "Amount": [100, 200, 50],
    })

    rfm = calculate_rfm(df)

    assert {"recency", "frequency", "monetary"}.issubset(rfm.columns)
