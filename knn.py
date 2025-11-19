from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer

explained_var = "value_eur"
explaination_vars = ["shooting", "passing", "dribbling", "defending"]

df = pd.read_csv("players_20.csv")

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

explaination_vars_full = explaination_vars + ["preferred_foot", "work_rate"] + list(positions_df.columns)
print(list(positions_df.columns))
df = df.dropna(subset=explaination_vars_full + [explained_var])

X = df[explaination_vars_full]
y = df[explained_var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)

y_pred = knn_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
