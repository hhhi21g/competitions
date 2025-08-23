import pandas as pd

train_labels = pd.read_parquet("dataset\\train_labels.parquet")

print(train_labels.head())
print(train_labels.columns)

candidates = pd.read_parquet("dataset\\candidates.parquet")

print(candidates.head())
print(candidates.columns)

train_dataset = pd.read_parquet("dataset\\train_dataset.parquet")

print(train_dataset.head())
print(train_dataset.columns)