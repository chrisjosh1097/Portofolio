### Sentiment Analysis with Deep Learning

This project focuses on building deep learning models to classify the sentiment of tweets using the 
[Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). We apply CNN, GRU, and Bi-LSTM 
architectures with Word2Vec embeddings to predict whether a tweet expresses a positive or negative sentiment.

---

## Dataset

- Size: 300,000+ labeled tweets  
- Features:
  - Sentiment (0 = Negative, 1 = Positive)
  - Tweet ID
  - Date
  - Query
  - Username
  - Tweet Text (primary feature for classification)

---

## Data Preprocessing

We performed a series of cleaning and preprocessing steps:
- Lowercasing
- URL and user mention removal
- Punctuation and special character removal
- Tokenization using `Word2Vec`
- Removed unnecessary features, focused on sentiment and text

*Stemming and lemmatization were not used to preserve sentiment-rich expressions.*



## Models Used

We evaluated three different deep learning models:
1. CNN (1D)
2. GRU (2-layer)
3. Bidirectional LSTM (Bi-LSTM)

All models use:
- Word2Vec embeddings
- Dropout layers to reduce overfitting
- Binary output via Sigmoid activation

## Hyperparameter Tuning

Grid search was applied to:
- Tune filters, kernel sizes, GRU/LSTM units, dense layers, dropout
- Prevent overfitting using early stopping

Tuning time:
- CNN: 148 mins
- GRU: 750 mins
- Bi-LSTM: 242 mins

## Results

Base Bi-LSTM performed best on all metrics in the Confusion Matrix.
---


