import pickle
import matplotlib.pyplot as plt
import seaborn as sns

paa_values = [0.03, 0.04, 0.05, 0.06, 0.07]

plt.figure(figsize=(12, 8))

for paa in paa_values:
    paasave = str(paa).replace('.', '')
    input_file = f"lt_hsc_distribution_paa_{paasave}.pkl"
    try:
        with open(input_file, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"File not found for PAA = {paa:.2f}. Skipping...")
        continue

    if paa in data:
        active_lt_hsc_values = data[paa]
    else:
        # If the key is not the float but a string, try to get the only value
        active_lt_hsc_values = list(data.values())[0]

    if not active_lt_hsc_values or len(set(active_lt_hsc_values)) < 2:
        print(f"Not enough data to plot KDE for PAA {paa}. Skipping...")
        continue

    kde = sns.kdeplot(active_lt_hsc_values, label=f"PAA = {paa:.2f}", linewidth=2)
    x, y = kde.get_lines()[-1].get_data()
    plt.fill_between(x, y, alpha=0.3)

plt.xlabel("Total LT-HSCs")
plt.ylabel("Density")
plt.title("KDEs of Total LT-HSCs for Different PAA Values")
plt.grid(axis='y', alpha=0.75)
plt.legend()
plt.xlim(left=0)
plt.savefig("kde_lt_hsc_paa_values.pdf", format="pdf", dpi=300)
plt.show()