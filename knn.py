import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from normalizer import normalize_dataframe_mixed

explained_var = "value_eur"
explaination_vars_full = [
    "overall", "international_reputation", "potential", "gk_handling", "movement_reactions", "age"
]
neighbors = 7

df = pd.read_csv("players_20.csv")
df =  normalize_dataframe_mixed(df)
df = df.dropna(subset=explaination_vars_full + [explained_var])

X = df[explaination_vars_full]
y = df[explained_var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_regressor = KNeighborsRegressor(n_neighbors=neighbors)
knn_regressor.fit(X_train, y_train)

y_pred = knn_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(explaination_vars_full)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
