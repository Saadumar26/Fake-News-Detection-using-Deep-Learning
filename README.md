#  Fake News Detection Using Deep Learning and NLP

## 📖 Introduction

In today's digital landscape, the spread of misinformation commonly referred to as **fake news** has become a major societal issue. The goal of this project is to build an intelligent system that can automatically detect and classify news articles as **fake** or **true**, using a combination of **Natural Language Processing (NLP)** and **Deep Learning** techniques.

This project not only explores classical machine learning algorithms but also fine-tunes a modern transformer model (**DistilBERT**) to achieve high performance. The system is deployed via a **Streamlit** web application, allowing users to test news headlines or content in real time.


## 🎯 Objectives

- Preprocess raw news text using advanced NLP techniques.
- Train and evaluate both traditional and deep learning models.
- Fine-tune a transformer-based model for binary classification.
- Visualize important text patterns and model performance.
- Deploy the system for real-time use through an interactive web interface.


## 🧰 Tools, Technologies, and Libraries

| Category               | Tools & Libraries Used                                                  |
|------------------------|-------------------------------------------------------------------------|
| **Language**           | Python                                                                  |
| **Preprocessing**      | SpaCy, Regular Expressions, Custom Lemmatization                        |
| **Feature Extraction** | TF-IDF Vectorizer                                                       |
| **Machine Learning**   | Logistic Regression (Scikit-learn)                                      |
| **Deep Learning**      | DistilBERT (`TFDistilBertForSequenceClassification`) via Hugging Face   |
| **Visualization**      | Matplotlib, Seaborn, WordCloud                                          |
| **Deployment**         | Streamlit                                                               |


## 🔄 Pipelines Overview

This project consists of two distinct pipelines: one for traditional ML and one for transformer-based deep learning.

### 🔹 1. Traditional ML Pipeline (Logistic Regression)

```
Raw News Text
     ↓
Text Cleaning & Preprocessing
  - Lowercasing
  - Punctuation & Stopword Removal
  - Lemmatization (SpaCy)
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression Model
     ↓
Prediction: Fake or True
```

### 🔹 2. Deep Learning Pipeline (DistilBERT)

```
Raw News Text
     ↓
Tokenization using DistilBertTokenizerFast
  - input_ids
  - attention_mask
     ↓
Fine-tuned DistilBERT Model
  (TFDistilBertForSequenceClassification)
     ↓
Softmax Output → Final Prediction
     ↓
Prediction: Fake or True
```


## 🧪 Workflow Overview

1. **Data Loading**: Importing and examining labeled news dataset (1 = True, 0 = Fake).
2. **Text Preprocessing**:
   - Lowercasing, punctuation removal, stopword removal
   - Lemmatization using SpaCy
3. **Feature Engineering**:
   - Classical ML: TF-IDF vectorization
   - Deep Learning: Tokenization with `DistilBertTokenizerFast`
4. **Model Training**:
   - **Logistic Regression** for a fast and interpretable baseline
   - **DistilBERT** fine-tuned on labeled news for high accuracy
5. **Evaluation**:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix and Classification Reports
6. **Visualization**:
   - Word clouds for fake vs. true news patterns
   - Graphs for performance comparison
7. **Deployment**:
   - Developed an interactive **Streamlit app** for user input and real-time predictions


## 📈 Results

- **Logistic Regression Accuracy**: 99.3%
- **DistilBERT Accuracy**: 99.8%
- The fine-tuned DistilBERT model outperformed traditional models in both precision and recall, making it the primary model for deployment.


## 🔮 Future Enhancements

- Extend to **multi-class classification** (e.g., satire, bias, opinion).
- Integrate **real-time news scraping APIs** for live predictions.
- Implement **ensemble models** to further boost performance.
- Add **explainability modules** (e.g., LIME/SHAP) for transparency.
- Deploy on **cloud platforms** for global accessibility.


> 💡 This project combines the power of traditional ML and modern transformer-based models to tackle one of the most pressing issues in digital communication: detecting fake news with high confidence and clarity.


**Developed by:** *M. Saad Umar*  
*Department of Information Technology, Bahauddin Zakariya University, Multan*
