from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# --------------------------------------------------
# 1. Load IMDb dataset
# --------------------------------------------------

dataset = load_dataset("stanfordnlp/imdb")

print(dataset)


# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

model_name = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)


# --------------------------------------------------
# 3. Tokenize the reviews
# --------------------------------------------------

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)


# --------------------------------------------------
# 4. Load pretrained DistilBERT
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# --------------------------------------------------
# 5. Training configuration
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="./sentiment-model",

    eval_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=100,

    save_strategy="epoch",

    load_best_model_at_end=True
)


# --------------------------------------------------
# 6. Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_dataset["train"],

    eval_dataset=tokenized_dataset["test"],

    tokenizer=tokenizer
)


# --------------------------------------------------
# 7. Train
# --------------------------------------------------

trainer.train()


# --------------------------------------------------
# 8. Save model
# --------------------------------------------------

trainer.save_model("./sentiment-model")

tokenizer.save_pretrained("./sentiment-model")

print("Model training complete!")  