import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Paramètres ===
binsNb = 3
step = 500_000

df = pd.read_csv("players_20.csv")

df = df.dropna(subset=['value_eur'])

# === Fonction de comptage ===
def get_nb(df, min_val, max_val):
    df_block = df[(df["value_eur"] >= min_val) & (df["value_eur"] < max_val)]
    return len(df_block)

counts = []
labels = []

for i in range(binsNb-1):
    min_value = i * step
    max_value = (i + 1) * step
    effective = get_nb(df, min_value, max_value)
    counts.append(effective)

    label_name = str(min_value / 1000)+"k - "+str(max_value / 1000) +"k"
    labels.append(label_name)

min_value = (binsNb-1) * step
label_name = ">"+str(min_value / 1000)+"k"
labels.append(label_name)
df_block = df[(df["value_eur"] >= min_value)]
counts.append(len(df_block))

result_df = pd.DataFrame({"Tranche (€M)": labels, "Nombre de joueurs": counts})

plt.figure(figsize=(14, 6))
sns.barplot(x="Tranche (€M)", y="Nombre de joueurs", data=result_df, color="skyblue")
plt.xticks(rotation=45, ha='right')
plt.title("Répartition des joueurs par tranche de valeur")
plt.xlabel("Valeur (en euros)")
plt.ylabel("Nombre de joueurs")
plt.tight_layout()
plt.show()
