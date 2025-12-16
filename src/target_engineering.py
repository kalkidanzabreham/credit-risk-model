# src/target_engineering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def calculate_rfm(
    df: pd.DataFrame,
    customer_col: str = "CustomerId",
    date_col: str = "TransactionStartTime",
    amount_col: str = "Amount",
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary (RFM) metrics per customer.
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby(customer_col)
        .agg(
            recency=(date_col, lambda x: (snapshot_date - x.max()).days),
            frequency=(date_col, "count"),
            monetary=(amount_col, "sum"),
        )
        .reset_index()
    )

    return rfm


def assign_risk_label(
    rfm_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster customers using RFM and assign high-risk label.
    """

    features = ["recency", "frequency", "monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[features])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    rfm_df["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster:
    # high recency (inactive), low frequency, low monetary
    cluster_summary = (
        rfm_df.groupby("cluster")[features]
        .mean()
        .sort_values(by=["frequency", "monetary"])
    )

    high_risk_cluster = cluster_summary.index[0]

    rfm_df["is_high_risk"] = (rfm_df["cluster"] == high_risk_cluster).astype(int)

    return rfm_df


def build_proxy_target(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full pipeline to generate is_high_risk target.
    """

    rfm = calculate_rfm(transactions_df)
    rfm_labeled = assign_risk_label(rfm)

    return rfm_labeled[["CustomerId", "is_high_risk"]]
