import pandas as pd

fake = pd.read_csv("data/raw/Fake.csv")
true = pd.read_csv("data/raw/True.csv")

fake["label"] = 0
true["label"] = 1

dataset = pd.concat(
    [fake, true],
    ignore_index=True
)

dataset["content"] = (
    dataset["title"].fillna("") +
    " " +
    dataset["text"].fillna("")
)

dataset = dataset[
    ["content", "label"]
]

dataset = dataset.sample(
    frac=1,
    random_state=42
)

dataset.to_csv(
    "data/processed/dataset_final.csv",
    index=False
)

print("Dataset criado com sucesso!")
print("Total de registros:", len(dataset))