import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Choose the PAA value you want to plot
paa_to_plot = 0.05
paasave = str(paa_to_plot).replace('.', '')

# Load the data from the corresponding pickle file
with open(f"lt_hsc_distribution_paa_{paasave}.pkl", "rb") as f:
    data = pickle.load(f)

# Extract the values (should be a dict with one key)
active_lt_hsc_values = list(data.values())[0]

plt.figure(figsize=(8, 6))
sns.histplot(
    active_lt_hsc_values,
    bins=30,
    binwidth=1,
    kde=True,
    color='skyblue',
    stat='density',
    label='Histogram',
    common_norm=True
)
sns.kdeplot(active_lt_hsc_values, color='red', linewidth=2, label='KDE')
plt.xlabel("Total LT-HSCs")
plt.ylabel("Density")
plt.title(f"Histogram and KDE for PAA = {paa_to_plot:.2f}")
plt.legend()
plt.xlim(left=0)
plt.grid(axis='y', alpha=0.75)
plt.savefig(f"hist_kde_paa_{paa_to_plot:.2f}.pdf", format="pdf", dpi=300)

plt.show()