import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.model_selection import train_test_split


# Paths & constants

BASE_MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path("./command_model")
DATASET_PATH = Path("dataset.json")

MAX_SEQUENCE_LENGTH = 64
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
EPOCHS = 8
BATCH_SIZE = 8


# Dataset loading

def load_json_dataset(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# Label processing

def build_label_index(data: list) -> Tuple[Dict[str, int], Dict[int, str]]:
    label_to_id = {}
    id_to_label = {}

    for index, item in enumerate(data):
        command = item["command"]
        label_to_id[command] = index
        id_to_label[index] = command

    return label_to_id, id_to_label


def extract_texts_and_labels(
    data: list,
    label_to_id: Dict[str, int]
) -> Tuple[List[str], List[int]]:
    texts = []
    labels = []

    for item in data:
        label_id = label_to_id[item["command"]]

        for example in item.get("examples", []):
            texts.append(example["text"])
            labels.append(label_id)

    return texts, labels


# Dataset

class CommandIntentDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)

        return item


# Model

def create_model(
    num_labels: int,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str]
):
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=num_labels
    )

    model.config.label2id = label_to_id
    model.config.id2label = id_to_label

    return model


# Training

def create_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_dir=str(OUTPUT_DIR / "logs"),
        load_best_model_at_end=True,
        report_to="none"
    )


def train():
    # Load data
    raw_data = load_json_dataset(DATASET_PATH)

    # Build labels
    label2id, id2label = build_label_index(raw_data)

    # Extract samples
    texts, labels = extract_texts_and_labels(raw_data, label2id)

    # Split dataset
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # Datasets
    train_dataset = CommandIntentDataset(
        train_texts,
        train_labels,
        tokenizer,
        MAX_SEQUENCE_LENGTH
    )

    test_dataset = CommandIntentDataset(
        test_texts,
        test_labels,
        tokenizer,
        MAX_SEQUENCE_LENGTH
    )

    # Model
    model = create_model(
        num_labels=len(label2id),
        label_to_id=label2id,
        id_to_label=id2label
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=create_training_args(),
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Train
    trainer.train()

    # Save artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


# Entry point

if __name__ == "__main__":
    train()
