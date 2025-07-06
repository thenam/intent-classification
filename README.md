# intent-classification
Intent classification in post and logistics

## Project Structure
```
intent-classification/
├── data/
│   ├── raw/                # Dữ liệu gốc
│   └── processed/          # Dữ liệu đã tiền xử lý
├── notebooks/              # Notebook Jupyter để thử nghiệm
├── src/
│   ├── data/               # Tiền xử lý dữ liệu
│   ├── models/             # Huấn luyện và đánh giá mô hình
│   ├── utils/
│   ├── predict.py          # Script dự đoán ý định từ văn bản
│   ├── evaluate.py         # Script đánh giá
├── models/                 # Mô hình đã finetune
├── reports/                # Báo cáo đánh giá mô hình
│   ├── figures/
│   └── metrics/
├── tests/                  # Kiểm thử đơn vị
├── requirements.txt
├── README.md
└── .gitignore

```

## Install
```
conda create -n intent-classification-vnp python=3.11
conda activate intent-classification-vnp
pip install -r requirements.txt 
```

## For Nvidia 50 Serial (use sm_120)
```
pip uninstall torch torchvision torchaudio
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Run
### Create train validation test data
```
python src/data/data_preprocessing.py
```

### Fine-tune model
```
python src/models/finetune_model.py
```

### Evaluate
```
python src/evaluate.py
```

### Predict
```
python src/ppredict.py
```

## Report
[Summary](reports/summary.md)