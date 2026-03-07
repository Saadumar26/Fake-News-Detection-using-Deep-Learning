# 📰 Fake News Detection using Deep Learning

A Natural Language Processing (NLP) project that detects whether a news article is **real or fake** using both classical machine learning and modern transformer-based deep learning models.

The project progressively explores four modeling approaches — starting from a traditional TF-IDF + Logistic Regression baseline and moving toward a fine-tuned DistilBERT transformer model.

🔗 **Live Demo:** [fake-news-detection-using-deep-learning-46mquyuftsbubrxddakfw8.streamlit.app](https://fake-news-detection-using-deep-learning-46mquyuftsbubrxddakfw8.streamlit.app/)

---

## 🧠 Project Objective

The rapid spread of misinformation on digital platforms makes automated fake news detection an important research problem. This project aims to build a system that can automatically classify news articles as **REAL** or **FAKE** using Natural Language Processing techniques.

To explore model performance across different levels of complexity, the project compares:

1. Classical machine learning models
2. Transformer-based contextual embeddings
3. Fully fine-tuned deep learning models

---

## 📂 Dataset

Two labeled CSV files were used as the source data:

| File | Label | Original Rows | After Cleaning |
|---|---|---|---|
| `True.csv` | 1 (Real) | 21,417 | 21,173 |
| `Fake.csv` | 0 (Fake) | 23,481 | 12,527 |
| **Combined** | — | — | **33,700** |

Cleaning steps applied:
- Removed duplicate rows and empty text entries
- Removed URLs, digits, and punctuation
- Applied **SpaCy** batch lemmatization and lowercasing

---

## ✂️ Train / Test Split

```
80% Training → 26,960 samples
20% Testing  →  6,740 samples
(Stratified split, random_state=42)
```

---

## 🤖 Models & Results

### Model 1 — Logistic Regression + TF-IDF (Default)

A classical NLP baseline using TF-IDF features (5,000 features, unigrams + bigrams).

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| FAKE | 0.98 | 0.98 | 0.98 |
| TRUE | 0.99 | 0.99 | 0.99 |
| **Overall Accuracy** | | | **99.00%** |
| **ROC-AUC** | | | **0.9989** |

---

### Model 2 — Logistic Regression + TF-IDF (Tuned C=10)

5-fold cross-validation was used to find the best regularization parameter `C`.

| C Value | Mean F1 |
|---|---|
| 0.01 | 0.9414 |
| 0.1 | 0.9727 |
| 1 | 0.9881 |
| **10** | **0.9932** ✅ |

Results with best `C=10`:

| Metric | Score |
|---|---|
| Test Accuracy | **99.18%** |
| ROC-AUC | **0.9994** |

---

### Model 3 — DistilBERT Embeddings + Logistic Regression

Instead of TF-IDF, 768-dimensional CLS token embeddings were extracted from a pre-trained DistilBERT model and fed into Logistic Regression.

| Metric | Score |
|---|---|
| Train Accuracy | 99.47% |
| Test Accuracy | **99.30%** |

---

### Model 4 — Fine-Tuned DistilBERT ⭐ Best Model

`DistilBertForSequenceClassification` was fine-tuned end-to-end on the dataset.

**Training Configuration:**

| Setting | Value |
|---|---|
| Max token length | 256 |
| Batch size | 32 |
| Epochs | 4 |
| Optimizer | AdamW (lr=1e-5) |
| Scheduler | Linear warmup (10%) |
| Early stopping | Patience = 2 |

**Training History:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.1061 | 96.04% | 0.0069 | 99.87% |
| 2 | 0.0036 | 99.93% | 0.0052 | 99.85% |
| 3 | 0.0016 | 99.96% | 0.0038 | 99.94% |
| 4 | 0.0005 | 99.99% | 0.0036 | **99.94%** |

**Final Evaluation:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| FAKE | 1.00 | 1.00 | 1.00 |
| TRUE | 1.00 | 1.00 | 1.00 |
| **Overall Accuracy** | | | **99.94%** |

Only **4 misclassifications** out of 6,732 samples.

```
Confusion Matrix:
[[2498    2]
 [   2 4230]]
```

---

## 📊 Model Comparison

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| LR + TF-IDF (Default) | 99.00% | 0.9989 |
| LR + TF-IDF (Tuned C=10) | 99.18% | 0.9994 |
| DistilBERT Embeddings + LR | 99.30% | — |
| **Fine-Tuned DistilBERT** | **99.94%** | — |

---

## 🌐 Web Application

The fine-tuned model is deployed as an interactive Streamlit app. Users can paste any news article and instantly get a prediction with a confidence score.

**Features:**
- Real-time FAKE / TRUE classification
- Confidence score with visual progress bar
- Word and character count display

🔗 **Try it live:** [fake-news-detection-using-deep-learning-46mquyuftsbubrxddakfw8.streamlit.app](https://fake-news-detection-using-deep-learning-46mquyuftsbubrxddakfw8.streamlit.app/)

The model weights are hosted on Hugging Face: [saadumar26/fake-news-detector](https://huggingface.co/saadumar26/fake-news-detector)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Classical ML | Scikit-learn |
| Text Processing | SpaCy, NLTK |
| Web App | Streamlit |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Training Environment | Google Colab (T4 GPU) |

---

## 📁 Repository Structure

```
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── data/
│   ├── True.csv                    # Real news dataset
│   └── Fake.csv                    # Fake news dataset
├── models/
│   └── distilbert_finetuned/       # Fine-tuned model config & tokenizer
├── artifacts/
│   └── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer
├── tokenizer/
│   └── saved_tokenizer/            # Saved tokenizer files
└── notebooks/
    └── Fake_News_Detection.ipynb   # Full training notebook
```

---

## ▶️ Run Locally

```bash
# Clone the repository
git clone https://github.com/Saadumar26/Fake-News-Detection-using-Deep-Learning.git
cd Fake-News-Detection-using-Deep-Learning

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🔮 Future Improvements

- **Multilingual support** — Extend to Urdu and other languages using mBERT or XLM-RoBERTa
- **Explainable AI** — Use LIME or SHAP to highlight which words influenced the prediction
- **Real-time news testing** — Integrate a news API so users can paste a URL directly
- **Multi-class classification** — Categorize news into satire, propaganda, misleading, etc.
- **Larger transformer models** — Experiment with RoBERTa or DeBERTa for even better performance

---

## 👤 Author

**Muhammad Saad Umar**


BS Information Technology — Bahauddin Zakariya University, Multan


Final Year Project — NLP & Deep Learning for Misinformation Detection
