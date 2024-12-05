import pandas as pd

# Load the feature and label files
features = pd.read_csv("features.csv")
labels = pd.read_csv("labels.csv")

# Merge datasets on the 'filename' column
merged = pd.merge(features, labels, on="filename")

# Save the merged dataset to a new CSV file
merged.to_csv("dataset.csv", index=False)
print("Merged dataset saved as dataset.csv")
