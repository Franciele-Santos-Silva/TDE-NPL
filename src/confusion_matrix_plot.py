import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

tokenizer = AutoTokenizer.from_pretrained("models")
model = AutoModelForSequenceClassification.from_pretrained("models")

model.eval()

df = pd.read_csv(
    "data/processed/dataset_final.csv"
)

df = df.sample(
    n=10000,
    random_state=42
)

_, test_texts, _, test_labels = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

predictions = []

for text in test_texts:

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    predictions.append(pred)

cm = confusion_matrix(
    test_labels,
    predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "True"],
    yticklabels=["Fake", "True"]
)

plt.title("Matriz de Confusão")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()
