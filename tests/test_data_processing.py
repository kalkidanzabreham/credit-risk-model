import pandas as pd
from src.data_processing import handle_missing_values


def test_handle_missing_values_fills_nulls():
    data = {
        "Amount": [100, None, 300],
        "Value": [100, 200, None],
        "ProductCategory": ["A", None, "B"]
    }

    df = pd.DataFrame(data)
    processed_df = handle_missing_values(df)

    assert processed_df.isnull().sum().sum() == 0
