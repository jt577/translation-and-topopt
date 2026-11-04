import pandas as pd

# --- load datasets ---
# dataset1 = pd.read_csv("data1_shearcond.csv")
# dataset2 = pd.read_csv("data2_shearcond.csv")
# dataset = pd.concat([dataset1, dataset2], ignore_index=True)
dataset = pd.read_csv("data_TPMS_shearcond.csv")

# --- compute objective ---
dataset['objective'] = 0.5 * dataset['conductivity'] + 0.5 * dataset['shear_modulus']

# --- find max objective value ---
max_val = dataset['objective'].max()

# --- rows achieving the maximum ---
max_rows = dataset[dataset['objective'] == max_val]

# --- print results ---
print("\nMaximum objective value:", max_val)

print("\nEntries achieving the maximum:")
print(max_rows)

# If your entry label column is called "name", extract just the names:
if 'name' in dataset.columns:
    names = max_rows['name'].tolist()
    print("\nNames of maximizing entries:")
    print(names)

# --- optional: top 10 ranking ---
print("\nTop 10 entries by objective:")
top10 = dataset.sort_values('objective', ascending=False).head(10)
print(top10)
