import numpy as np
import pandas as pd
from normalizer import normalize_dataframe_mixed

df = pd.read_csv("players_20.csv")
df = normalize_dataframe_mixed(df)

numeric_df = df.select_dtypes(include=[np.number])

excluded_cols = [
    "value_eur",
    "release_clause_eur",
    "wage_eur",
    "sofifa_id",
    "team_jersey_number",
    "nation_jersey_number"
]

clean_df = numeric_df.drop(columns=[c for c in excluded_cols if c in numeric_df.columns])


corr = clean_df.corrwith(df["value_eur"]).sort_values(ascending=False)

print(corr)