import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

ALL_DATA_PATH = "data/raw/all_data/intent_data.jsonl"

def clean_text(text):
    
    # Chuyen ve chu thuong
    text = text.lower()
    # Bo dau cau
    text = re.sub(r'[^\w\s]', '', text)
    # Bo ky tu \n \t
    text = re.sub(r'[\n\t]+', ' ', text).strip()

    return text

def split_data(df):

    # Tach lay tap train
    train_df, val_test_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df["label"],
        random_state=42
    )

    # Tach lay tap validation, test
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=0.5,
        stratify=val_test_df["label"],
        random_state=42
    )

    return train_df, val_df, test_df

df = pd.read_json(ALL_DATA_PATH, lines=True)

df["clean_text"] = df["text"].apply(clean_text)

train_df, val_df, test_df = split_data(df)

# doi ten cot
train_df = train_df[["text", "label", "clean_text"]].rename(columns={"text": "text_origin"}).rename(columns={"clean_text": "text"})
val_df = val_df[["text", "label", "clean_text"]].rename(columns={"text": "text_origin"}).rename(columns={"clean_text": "text"})
test_df = test_df[["text", "label", "clean_text"]].rename(columns={"text": "text_origin"}).rename(columns={"clean_text": "text"})

# danh sach cac column can ghi vao file JSONL
columns_to_save = ["text", "label"]

# ghi file
train_df[columns_to_save].to_json('data/processed/train.jsonl', orient='records', lines=True, force_ascii=False)
val_df[columns_to_save].to_json('data/processed/validation.jsonl', orient='records', lines=True, force_ascii=False)
test_df[columns_to_save].to_json('data/processed/test.jsonl', orient='records', lines=True, force_ascii=False)

# tao label file
label2id = {
    "bccp_nd": 0,
    "bccp_qt": 1,
    "tcbc": 2,
    "ppbl": 3,
    "hcc": 4,
    "cccs": 5,
    "dtcp": 6,
}

with open('data/processed/label2id.json', 'w') as f:
    f.write(json.dumps(label2id, indent=4))