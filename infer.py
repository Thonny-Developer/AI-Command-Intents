import re
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Constants

MODEL_DIR = Path("./command_model")
MAX_SEQUENCE_LENGTH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Regex patterns

TIME_PATTERN = re.compile(r"(\d{1,2}[:.]?\d{0,2})\s*(утра|вечера|am|pm)?", re.IGNORECASE)
VOLUME_PATTERN = re.compile(r"(\d{1,3})\s*%", re.IGNORECASE)
WEBSITE_PATTERN = re.compile(
    r"(ютуб|youtube|реддит|reddit|твитч|twitch|роблокс|roblox|почту|gmail|mail|тикток|tiktok)",
    re.IGNORECASE
)
TIMER_PATTERN = re.compile(r"(\d+)\s*(секунд|минут|часов)?", re.IGNORECASE)


# Command processor

class CommandProcessor:
    def __init__(self, model_dir: Path):
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        self.model.to(DEVICE)
        self.model.eval()

    def predict_command(self, text: str) -> str:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        )

        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits

        label_id = torch.argmax(logits, dim=1).item()
        return self.model.config.id2label[label_id]

    def extract_arguments(self, command: str, text: str) -> Dict[str, Any]:
        args: Dict[str, Any] = {}

        if command == "set_alarm":
            match = TIME_PATTERN.search(text)
            if match:
                time = match.group(1).replace(".", ":")
                period = match.group(2)
                if period:
                    time = f"{time} {period.lower()}"
                args["time"] = time

        elif command == "set_volume":
            match = VOLUME_PATTERN.search(text)
            if match:
                args["volume"] = f"{match.group(1)}%"

        elif command == "open_website":
            match = WEBSITE_PATTERN.search(text)
            if match:
                args["website"] = match.group(1).lower()

        elif command == "set_timer":
            match = TIMER_PATTERN.search(text)
            if match:
                unit = match.group(2) if match.group(2) else "секунд"
                args["duration"] = f"{match.group(1)} {unit}"

        return args

    def process(self, text: str) -> Dict[str, Any]:
        command = self.predict_command(text)
        arguments = self.extract_arguments(command, text)
        return {
            "command": command,
            "arguments": arguments
        }


# Entry point

def main():
    processor = CommandProcessor(MODEL_DIR)

    while True:
        user_input = input("Введите запрос: ").strip()
        if not user_input:
            continue

        result = processor.process(user_input)
        print(result)


if __name__ == "__main__":
    main()
