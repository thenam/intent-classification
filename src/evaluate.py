import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import matplotlib.pyplot as plt
import seaborn as sns

with open("data/processed/label2id.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("models/vnp-intent", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("models/vnp-intent", num_labels=len(label2id))

test_dataset = load_dataset("json", data_files={"test": "data/processed/test.jsonl"})["test"]

def encode_label(data):
    data["label"] = label2id[data["label"]]
    return data
test_dataset = test_dataset.map(encode_label)

def tokenize(data):
    return tokenizer(data["text"], padding="max_length", truncation=True, max_length=128)
tokenized_test_dataset = test_dataset.map(tokenize, batched=True)

trainer = Trainer(model=model, tokenizer=tokenizer)

predictions = trainer.predict(tokenized_test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

# Classification Report
target_names = [label for _, label in sorted((v, k) for k, v in label2id.items())]
report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

print("\n Classification Report:")
print(report)

classification_report_path = "reports/metrics/classification_report.txt"
with open(classification_report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Report saved to {classification_report_path}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()

confusion_matrix_path = "reports/metrics/confusion_matrix.png"
plt.savefig(confusion_matrix_path)
print(f"Confusion matrix image saved to {confusion_matrix_path}")
plt.close()