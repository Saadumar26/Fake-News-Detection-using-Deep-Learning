# 📰 Fake News Detection using Deep Learning

A machine learning project that detects whether a news article is **real or fake** — using a combination of classical ML and modern deep learning approaches. The project progressively builds from a simple TF-IDF + Logistic Regression baseline all the way up to a fine-tuned DistilBERT transformer model.

---

## 🧠 What This Project Does

The goal is straightforward: given a news article's text, can we automatically tell if it's real or fake? This project explores that question by training and comparing four different models, each more sophisticated than the last. Spoiler: the fine-tuned transformer basically nails it.

---

## 📂 Dataset

The project uses two CSV files — `True.csv` and `Fake.csv` — which are labeled accordingly:

- `True.csv` → label `1` (Real news)
- `Fake.csv` → label `0` (Fake news)

After merging, the final cleaned dataset contains **33,700 articles**.

---

## 🧹 Data Preprocessing

Before training anything, the raw text goes through a solid cleaning pipeline:

- Removed duplicate rows from both `True.csv` and `Fake.csv` separately
- Dropped rows with empty or null text values
- Applied **basic cleaning**: removed URLs, digits, and punctuation
- Used **SpaCy** for batch lemmatization and lowercasing
- Final combined DataFrame: `33,700 rows × 5 columns`

The cleaned text was saved as `cleaned_text_with_token_reduction.csv` for reuse across models.

---

## ✂️ Train / Test Split

```python
train_test_split(..., test_size=0.2, stratify=df['label'], random_state=42)
```

| Set        | Size   |
|------------|--------|
| Training   | 26,960 |
| Testing    | 6,740  |

---

## 🔢 Vectorization (for classical models)

Used **TF-IDF** to convert text into numerical features:

```python
TfidfVectorizer(max_features=5000, ngram_range=(1,2), max_df=0.95, min_df=2)
```

- Training matrix shape: `(26,960 × 5,000)`
- Testing matrix shape: `(6,740 × 5,000)`

---

## 🤖 Models

### Model 1 — Logistic Regression (TF-IDF, Default C=1)

A simple baseline to see how far classic ML can take us.

| Metric     | FAKE | TRUE |
|------------|------|------|
| Precision  | 0.98 | 0.99 |
| Recall     | 0.98 | 0.99 |
| F1-Score   | 0.98 | 0.99 |
| **Overall Accuracy** | **0.99** | |
| **ROC-AUC Score**    | **0.9989** | |

Confusion Matrix:
```
[[2459   46]
 [  42 4193]]
```

---

### Model 2 — Logistic Regression (TF-IDF, Tuned C=10)

Cross-validation was done across `C = [0.01, 0.1, 1, 10]` using 5-fold CV. Best C was **10**.

| C Value | Mean F1 | Std Dev |
|---------|---------|---------|
| 0.01    | 0.9414  | 0.0058  |
| 0.1     | 0.9727  | 0.0046  |
| 1       | 0.9881  | 0.0019  |
| **10**  | **0.9932** | **0.0006** |

Results with `C=10`:

| Metric     | FAKE | TRUE |
|------------|------|------|
| Precision  | 0.99 | 0.99 |
| Recall     | 0.99 | 0.99 |
| F1-Score   | 0.99 | 0.99 |
| **Overall Accuracy** | **0.99** | |
| **ROC-AUC Score**    | **0.9994** | |

```
Train Accuracy: 0.9991
Test Accuracy:  0.9918
```

---

### Model 3 — DistilBERT Embeddings + Logistic Regression

Instead of TF-IDF, this model extracts **768-dimensional CLS embeddings** from a pre-trained DistilBERT model and feeds them into Logistic Regression.

- Total DistilBERT parameters: `66,362,880`
- Embedding extraction batch size: 128, max_length: 128

Embedding shapes:
```
X_train_embeddings: (26,939, 768)
X_test_embeddings:  (6,735, 768)
```

| Metric     | FAKE | TRUE |
|------------|------|------|
| Precision  | 0.99 | 0.99 |
| Recall     | 0.99 | 0.99 |
| F1-Score   | 0.99 | 0.99 |
| **Overall Accuracy** | **0.99** | |

```
Train Accuracy: 0.9947
Test Accuracy:  0.9930
```

---

### Model 4 — Fine-Tuned DistilBERT (End-to-End)

This is the main event. `DistilBertForSequenceClassification` is fine-tuned directly on the news dataset with full backpropagation.

**Training Configuration:**

| Setting          | Value               |
|------------------|---------------------|
| Max token length | 256                 |
| Batch size       | 32                  |
| Epochs           | 4                   |
| Optimizer        | AdamW (lr=1e-5)     |
| Scheduler        | Linear warmup (10%) |
| Loss function    | CrossEntropyLoss (class-weighted) |
| Early stopping   | Patience = 2        |
| Train / Val size | 26,920 / 6,732      |

**Training History:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc  |
|-------|------------|-----------|----------|----------|
| 1     | 0.1061     | 0.9604    | 0.0069   | 0.9987   |
| 2     | 0.0036     | 0.9993    | 0.0052   | 0.9985   |
| 3     | 0.0016     | 0.9996    | 0.0038   | 0.9994   |
| 4     | 0.0005     | 0.9999    | 0.0036   | **0.9994** |

**Final Evaluation:**

| Metric     | FAKE | TRUE |
|------------|------|------|
| Precision  | 1.00 | 1.00 |
| Recall     | 1.00 | 1.00 |
| F1-Score   | 1.00 | 1.00 |
| **Overall Accuracy** | **1.00** | |

Confusion Matrix:
```
[[2498    2]
 [   2 4230]]
```

Only **4 misclassifications** out of 6,732 samples. That's about as good as it gets.

---

## 📊 Model Comparison Summary

| Model                              | Test Accuracy | ROC-AUC |
|------------------------------------|---------------|---------|
| LR + TF-IDF (C=1)                  | 0.99          | 0.9989  |
| LR + TF-IDF (Tuned C=10)           | 0.9918        | 0.9994  |
| DistilBERT Embeddings + LR         | 0.9930        | —       |
| **Fine-Tuned DistilBERT**          | **~1.00**     | —       |

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data handling
- **SpaCy** — Text preprocessing & lemmatization
- **NLTK** — Stopwords, WordNet
- **Scikit-learn** — TF-IDF, Logistic Regression, metrics
- **HuggingFace Transformers** — DistilBERT tokenizer & model
- **PyTorch** — Model training, GPU acceleration
- **Matplotlib / Seaborn** — Visualizations
- **tqdm** — Progress bars
- **Google Colab + Drive** — Runtime environment

---

## 📁 File Structure

Everything lives inside a single `FYP/` folder on Google Drive:

```
My Drive/FYP/
│
├── 📄 Raw Data
│   ├── True.csv                               # Original real news articles (21,417 rows)
│   └── Fake.csv                               # Original fake news articles (23,481 rows)
│
├── 🧹 Cleaned Data
│   ├── Fake_cleaned.csv                       # Cleaned fake news (after dedup + null removal)
│   ├── cleaned_data.csv                       # Full merged & cleaned dataset
│   └── cleaned_text_with_token_reduction.csv  # Final preprocessed dataset used for all models
│
├── 🏷️ Labels
│   └── labels.csv                             # Extracted label column (0 = FAKE, 1 = TRUE)
│
├── 🔢 TF-IDF Artifacts
│   ├── tfidf_vectorizer.pkl                   # Fitted TF-IDF vectorizer (for inference)
│   └── tfidf_matrix.pkl                       # Transformed TF-IDF feature matrix
│
├── 🤖 Saved Models
│   ├── final_logistic_regression_model.pkl    # Early LR model save
│   ├── logistic_regression_model.pkl          # LR model (tuned, C=10)
│   ├── fake_news_logreg_model.joblib          # LR model trained on BERT embeddings
│   └── trained_logistic_regression.pkl        # Final LR classifier (BERT embeddings version)
│
├── 🗂️ Saved Tokenizers
│   ├── fake_news_tokenizer/                   # Tokenizer for embedding-based LR model
│   └── saved_tokenizer/                       # Alternate tokenizer save
│
├── 🧠 Fine-Tuned DistilBERT
│   └── final_distilbert_model/                # Full fine-tuned model + tokenizer
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       └── ...
│
└── 📓 fake_news_detection.ipynb               # Main Colab notebook
```

---

## 🏗️ DistilBERT Model Architecture

The base `DistilBertModel` used for embedding extraction has this structure:

```
DistilBertModel(
  (embeddings):
    - word_embeddings:      Embedding(30522, 768)
    - position_embeddings:  Embedding(512, 768)
    - LayerNorm + Dropout

  (transformer): 6 × TransformerBlock, each with:
    - DistilBertSelfAttention  (Q, K, V, Out linear layers — 768→768)
    - LayerNorm after attention
    - FFN: Linear(768→3072) → GELU → Linear(3072→768)
    - LayerNorm after FFN
)
```

For fine-tuning, `DistilBertForSequenceClassification` adds a classification head on top:

```
(pre_classifier): Linear(768 → 768)
(classifier):     Linear(768 → 2)       ← binary: FAKE or TRUE
(dropout):        Dropout(p=0.2)
```

Total parameters: **66,362,880** (all trainable)

> **Note on load warnings:** When loading `distilbert-base-uncased` for classification, you'll see `UNEXPECTED` keys like `vocab_projector` and `MISSING` keys like `classifier.weight`. This is completely normal — those MLM head weights aren't needed for classification, and the new classification head gets freshly initialized before fine-tuning.

---

## ✅ Pre-Viva Sanity Tests

Before submission, a full suite of sanity checks was run to make sure everything was working end-to-end:

| Test | Result |
|------|--------|
| No NaNs in `clean_text` | ✅ Passed |
| Labels correctly mapped to 0/1 | ✅ Passed |
| Tokenization shape matches `MAX_LEN=256` | ✅ Passed |
| Tiny batch overfit test (loss: 0.6242) | ✅ Passed |
| Validation accuracy above 90% | ✅ Passed |
| Model reload from saved checkpoint | ✅ Passed |
| Inference on new sample inputs | ✅ Passed |

```
🚀 Running final sanity tests before Viva...
✅ Dataset test passed: no NaNs, labels mapped 0/1.
✅ Tokenization test passed: shapes match.
✅ Tiny batch overfit test passed: final loss 0.6242
✅ Validation accuracy above 90%.
✅ Reload test: Predictions on sample: [0 0]

🎉 All tests passed. Ready for Viva!
```

---

## ▶️ How to Run

### Option 1 — Load Data from GitHub (Recommended)

The dataset files (`True.csv` and `Fake.csv`) are available directly in this repository. You can load them in Colab without any manual uploading:

```python
import pandas as pd

base_url = "https://raw.githubusercontent.com/Saadumar26/Fake-News-Detection-using-Deep-Learning/main/"

true_df = pd.read_csv(base_url + "True.csv")
fake_df = pd.read_csv(base_url + "Fake.csv")
```

Then:
1. Open the notebook in Google Colab
2. Use the URLs above instead of local file paths
3. Run all cells in order — each section is clearly labeled
4. For fine-tuned DistilBERT training, enable GPU runtime (Runtime → Change runtime type → T4 GPU)

---

### Option 2 — Use Google Drive

The dataset is also shared via Google Drive. You can either download and re-upload it to your own Drive, or mount it directly:

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

true_df = pd.read_csv('/content/drive/My Drive/FYP/True.csv')
fake_df = pd.read_csv('/content/drive/My Drive/FYP/Fake.csv')
```

Then:
1. Upload `True.csv` and `Fake.csv` to your Drive under `My Drive/FYP/`
2. Open the notebook in Google Colab and mount Drive
3. Run all cells in order
4. For fine-tuned DistilBERT training, enable GPU runtime (Runtime → Change runtime type → T4 GPU)

---

## 📌 Notes

- Data leakage protection was applied before fine-tuning DistilBERT — any text samples that appeared in training were removed from the validation set
- Class weights were computed from the training set only to avoid leakage
- Early stopping was used to prevent overfitting (patience = 2)
- The best model checkpoint is restored after training completes
- All models and vectorizers are saved as `.pkl` / `.joblib` files for easy reuse without retraining

---

## 🔮 Future Improvements

There's definitely room to take this project further. Here are some directions worth exploring:

- **Try larger transformer models** — Something like RoBERTa, BERT-large, or DeBERTa could squeeze out even better performance, especially on more noisy or domain-specific news data
- **Multilingual support** — The current model only handles English. Extending it to Urdu or other languages using models like `mBERT` or `XLM-RoBERTa` would make it much more broadly useful
- **Explainability** — Adding LIME or SHAP to highlight which words/phrases actually pushed the model toward FAKE or TRUE would make the predictions more trustworthy and interpretable
- **Real-time news testing** — Integrating a news API (like NewsAPI) so users can paste a URL or headline and get a live prediction
- **Bigger and more diverse dataset** — Training on data from multiple sources and time periods would make the model more robust and less likely to overfit to a particular writing style
- **Confidence scoring** — Instead of just FAKE/TRUE, showing a confidence percentage would give users a better sense of how certain the model is
- **Streamlit Web App** — Deploying the model as an interactive web app on Streamlit Community Cloud so anyone can test it without needing to run a single line of code *(in progress)*

---

## 👤 Author

**Muhammad Saad Umar**
BS Information Technology
Bahauddin Zakariya University, Multan

This project was developed as part of a Final Year Project (FYP) that focuses on NLP and deep learning for misinformation detection.
