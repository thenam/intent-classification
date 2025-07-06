from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score

# Disable warning
#os.environ["WANDB_DISABLED"] = "true"

label2id = json.load(open("data/processed/label2id.json"))

train_dataset = load_dataset("json", data_files={"train": "data/processed/train.jsonl"})["train"]
val_dataset = load_dataset("json", data_files={"validation": "data/processed/validation.jsonl"})["validation"]

def encode_label(data):
    data["label"] = label2id[data["label"]]
    return data

train_dataset = train_dataset.map(encode_label)
val_dataset = val_dataset.map(encode_label)

def compute_metrics(eval_pred):
    '''
    Tính toán Accuracy & F1 trên tập validation
    '''

    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

def tokenize(data):
    return tokenizer(data["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train_dataset = train_dataset.map(tokenize, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=len(label2id)
)

training_args = TrainingArguments(
    output_dir="models",
    eval_strategy="epoch",
    save_strategy="best",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="models/logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# Luu model va tokenizer
print("Save model and tokenizer!")
trainer.save_model("models/vnp-intent")
tokenizer.save_pretrained("models/vnp-intent")

# Danh gia mo hinh
print("Evaluation results on validation set!")
eval_results = trainer.evaluate()
print("Danh gia mo hinh tren tap validation:")
print(eval_results)