import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # <-- ici
from sklearn.metrics import mean_squared_error, r2_score
from normalizer import normalize_dataframe_mixed

explained_var = "value_eur"
explaination_vars_full = [
    #"age", "overall"
    "movement_reactions", "age", "skill_ball_control", "defending_standing_tackle", "mentality_composure", "international_reputation", "attacking_finishing"
    #"age", "international_reputation", "attacking_crossing","attacking_finishing","attacking_heading_accuracy","attacking_short_passing","attacking_volleys","skill_dribbling","skill_curve","skill_fk_accuracy","skill_long_passing","skill_ball_control","movement_acceleration","movement_sprint_speed","movement_agility","movement_reactions","movement_balance","power_shot_power","power_jumping","power_stamina","power_strength","power_long_shots","mentality_aggression","mentality_interceptions","mentality_positioning","mentality_vision","mentality_penalties","mentality_composure","defending_marking","defending_standing_tackle","defending_sliding_tackle","goalkeeping_diving","goalkeeping_handling","goalkeeping_kicking","goalkeeping_positioning","goalkeeping_reflexes"
]

df = pd.read_csv("players_20.csv")
df = normalize_dataframe_mixed(df)

df.to_csv("normalized.csv")
df = df.dropna(subset=explaination_vars_full + [explained_var])

X = df[explaination_vars_full]
y = df[explained_var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_regressor.fit(X_train, y_train)

result = permutation_importance(
    rf_regressor, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

importances = pd.Series(result.importances_mean, index=X.columns)
print(importances.sort_values(ascending=False))
importances.sort_values().plot(kind="barh", figsize=(8,12))

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(explaination_vars_full)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
