import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import Dataset

from data_loader import load_dataset
from preprocess import preprocess_dataframe
from utils import ensure_dirs


def main():
    print("Carregando dataset...")

    ensure_dirs()
    set_seed(42)

    import os
    model_dir = "models/bert"
    test_split_path = "data/processed/test_split.csv"

    # Se já houver modelo e split de teste, evita re-treinar/tudo de novo.
    if os.path.exists(model_dir) and os.path.exists(test_split_path):
        print("Modelo e test_split.csv já existem. Pulando treinamento.")
        return

    df = load_dataset()
    df = preprocess_dataframe(df)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["clean_text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )

    # Salvar o split de teste preprocessado para evitar data leakage na avaliação
    # (usa exatamente os textos/labels do split criado acima)
    test_df = pd.DataFrame({
        "clean_text": test_texts,
        "label": test_labels
    })
    test_df.to_csv("data/processed/test_split.csv", index=False)


    print("Tokenizando...")

    MODEL_NAME = "prajjwal1/bert-tiny"

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=256
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=256
    )

    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels
    })

    test_dataset = Dataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": test_labels
    })

    print("Carregando BERT...")

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="outputs",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=500,
        logging_steps=100,
        seed=42,
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    print("Treinando...")

    trainer.train()

    print("Salvando modelo...")

    model.save_pretrained(
        "models/bert"
    )

    tokenizer.save_pretrained(
        "models/bert"
    )

    print("Treinamento concluído!")


if __name__ == "__main__":
    main()