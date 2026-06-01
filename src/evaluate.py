import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

print("Carregando modelo...")

tokenizer = AutoTokenizer.from_pretrained(
    "models"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "models"
)

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

accuracy = accuracy_score(
    test_labels,
    predictions
)

precision = precision_score(
    test_labels,
    predictions
)

recall = recall_score(
    test_labels,
    predictions
)

f1 = f1_score(
    test_labels,
    predictions
)

cm = confusion_matrix(
    test_labels,
    predictions
)

print("\nRESULTADOS")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\nMatriz de Confusão")
print(cm)

print("\nClassification Report")
print(
    classification_report(
        test_labels,
        predictions
    )
)
