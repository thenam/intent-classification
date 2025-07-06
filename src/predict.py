import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load mapping
with open("data/processed/label2id.json", "r") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# Load pre-train model de test
phobert_model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=len(label2id),
)

phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
phobert_classifier = pipeline("text-classification", model=phobert_model, tokenizer=phobert_tokenizer)

# Load fine-tuned model de test
vnp_tokenizer = AutoTokenizer.from_pretrained("models/vnp-intent", use_fast=False)
vnp_classifier = pipeline("text-classification", model="models/vnp-intent", tokenizer=vnp_tokenizer)

def predict(query, id2label, classifier):
    result = classifier(query)[0]
    label_id = int(result["label"].replace("LABEL_", ""))
    true_label = id2label[label_id]

    return {
            "label": true_label,
            "score": result["score"]
    }

queries = [
    "Doanh thu tháng 5 năm 2024 là bao nhiêu?",
    "Chuyển 1kg hàng từ HN vào TP HCM mất bao lâu",
    "Phân tích doanh thu quý 1 năm 2025 so với cùng kỳ",
    "thông tin bưu gửi T329024012",
    "Làm ơn hãy cho tôi biết doanh thu quý 1 năm ngoái của Bưu điện TP Hà Nội",
    "Hướng dẫn mua bảo hiểm",
    "Đăng ký vneid",
    "Nhận lương hưu thế nào"
]

for query in queries:
    print(query, "|- phobert", predict(query, id2label, phobert_classifier), "|- vnp", predict(query, id2label, vnp_classifier))