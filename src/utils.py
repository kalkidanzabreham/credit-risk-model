def validate_dataframe(df):
    """
    Basic validation to ensure the dataframe is not empty.
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty or None")
