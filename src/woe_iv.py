from xverse.transformer import WOE


def apply_woe(df, target_col, features):
    """
    Apply Weight of Evidence (WoE) transformation.
    """
    woe = WOE()
    woe.fit(df[features], df[target_col])

    woe_df = woe.transform(df[features])
    woe_df.columns = [f"woe_{col}" for col in woe_df.columns]

    return woe_df
