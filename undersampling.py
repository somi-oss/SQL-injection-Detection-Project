import pandas as pd

# Load dataset
df = pd.read_csv("Modified_SQL_Dataset.csv")

# Separate classes
benign = df[df["Label"] == 0]
sqli = df[df["Label"] == 1]

# Undersample benign class
benign_downsampled = benign.sample(
    n=len(sqli),
    random_state=42
)

# Combine and shuffle
balanced_df = pd.concat([benign_downsampled, sqli])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save new dataset
balanced_df.to_csv("Balanced_SQL_Dataset.csv", index=False)

print(balanced_df["Label"].value_counts())
