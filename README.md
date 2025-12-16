```md
# AI Command Intent Model

This repository contains an intent classification model used as the core of an AI voice assistant
The model receives free-form user text, predicts the intended command, and extracts structured arguments for further execution

The goal of this project is not just experimentation, but building a stable and extensible foundation for a production-ready assistant


## What This Project Does

- Classifies user input into predefined commands (intents)
- Maps natural language to structured command names
- Extracts arguments such as time, volume, duration, or target service
- Designed to be used in CLI tools, APIs, or real-time voice assistants
- Clean separation between training and inference logic


## Project Structure

```

.
├── dataset.json          # Training dataset (commands and examples)
├── train.py              # Model training pipeline
├── infer.py              # Inference / command processing logic
├── command_model/        # Trained model artifacts
└── README.md

````


## Model Overview

- Base model: DistilBERT
- Task: Sequence classification (intent detection)
- Framework: PyTorch + Hugging Face Transformers
- Output: Command label + extracted arguments

DistilBERT was chosen as a balance between accuracy, speed, and resource usage, making it suitable for real-time assistants.


## Installation

Install required dependencies:

```bash
pip install torch transformers scikit-learn
````

Optional but recommended for training progress visualization:

```bash
pip install tqdm
```

## Training the Model

Prepare your dataset in `dataset.json`, then run:

```bash
python train.py
```

After training, the model and tokenizer will be saved to the `command_model/` directory.
These artifacts are used directly by the inference layer.

## Running Inference

To test the model locally:

```bash
python infer.py
```

Example output:

```json
{
  "command": "set_alarm",
  "arguments": {
    "time": "7:30 am"
  }
}
```

## Design Principles

* Explicit code over magic
* Clear separation of responsibilities
* Predictable behavior in production
* Easy to extend with new commands or arguments
* Ready for API or microservice integration

## Future Improvements

* Confidence thresholds and fallback intents
* Advanced argument normalization
* Multilingual support
* Model versioning and dataset validation
* Metrics reporting (accuracy, F1 score)
* Deployment via FastAPI or gRPC

## Notes

This repository focuses on intent understanding, not speech recognition or TTS.
It is designed to be one component in a larger voice assistant system.

---

This project is intentionally kept simple and explicit.
Simple systems survive longer.
