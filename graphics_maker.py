import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Paramètres ===
var_name = "value_eur"
ranges = True
binsNb = 20
step = 500_000
starting_value = 0

df = pd.read_csv("players_20.csv")

df = df.dropna(subset=[var_name])

# === Fonction de comptage ===
def get_nb(df, min_val, max_val):
    if(ranges):
        df_block = df[(df[var_name] >= min_val) & (df[var_name] < max_val)]
    else:
        df_block = df[df[var_name] == min_val]

    return len(df_block)

counts = []
labels = []

for i in range(binsNb-1):
    min_value = starting_value + i * step
    max_value = starting_value + (i + 1) * step
    effective = get_nb(df, min_value, max_value)
    counts.append(effective)

    if(ranges):
        label_name = str(min_value)+" - "+str(max_value)
    else:
        label_name = str(min_value)
    labels.append(label_name)

min_value = starting_value + (binsNb-1) * step

if(ranges):
    label_name = ">"+str(min_value)
else:
    label_name = str(min_value)
labels.append(label_name)

df_block = df[(df[var_name] >= min_value)]
counts.append(len(df_block))

result_df = pd.DataFrame({"Tranche (€M)": labels, "Nombre de joueurs": counts})

plt.figure(figsize=(14, 6))
sns.barplot(x="Tranche (€M)", y="Nombre de joueurs", data=result_df, color="skyblue")
plt.xticks(rotation=45, ha='right')
plt.title("Répartition des joueurs par "+var_name)
plt.xlabel(var_name)
plt.ylabel("Nombre de joueurs")
plt.tight_layout()
plt.show()
