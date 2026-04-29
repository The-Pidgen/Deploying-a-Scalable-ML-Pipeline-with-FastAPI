import pandas as pd
from ml.data import process_data

df = pd.read_csv("data/census.csv")
print(df.head())
print(df.columns)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]
X, y, encoder, lb = process_data(
    df,
    categorical_features=cat_features,
    label="salary",
    training=True
)

print(X.shape)
print(y.shape)
print(type(encoder))
print(type(lb))