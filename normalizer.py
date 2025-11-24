import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def normalize_dataframe_mixed(df, num_mode="minmax", cat_mode="onehot"):
    """
    Normalise un DataFrame mixte avec colonnes numériques et catégorielles.

    Paramètres :
        df (pd.DataFrame) : DataFrame d'entrée
        num_mode (str) : 'minmax' ou 'zscore' pour colonnes numériques
        cat_mode (str) : 'onehot', 'label' ou 'frequency' pour colonnes catégorielles

    Retour :
        pd.DataFrame : DataFrame transformé
    """
    
    df = df.copy()
    
    # Colonnes numériques
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if num_mode == "minmax":
            min_val, max_val = df[col].min(), df[col].max()
            df[col] = 0 if max_val == min_val else (df[col] - min_val) / (max_val - min_val)
        elif num_mode == "zscore":
            mean, std = df[col].mean(), df[col].std()
            df[col] = 0 if std == 0 else (df[col] - mean) / std
        else:
            raise ValueError("num_mode doit être 'minmax' ou 'zscore'")
    
    # Colonnes catégorielles
    df["preferred_foot"] = df["preferred_foot"].map({"Left": 0, "Right": 1})
    df = df.dropna(subset=["preferred_foot"])

    df["work_rate"] = df["work_rate"].map({
        "Low/Low": 0,
        "Low/Medium": 1,
        "Medium/Low": 1,
        "Medium/Medium": 2,
        "High/Medium": 3,
        "Medium/High": 3,
        "High/High": 4
    })
    df = df.dropna(subset=["work_rate"])

    df["player_positions"] = df["player_positions"].str.split(r",\s*")

    mlb = MultiLabelBinarizer()
    positions_encoded = mlb.fit_transform(df["player_positions"])

    positions_df = pd.DataFrame(
        positions_encoded,
        columns=mlb.classes_,
        index=df.index
    )

    df = pd.concat([df, positions_df], axis=1)
    
    return df