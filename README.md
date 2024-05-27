# Sentiment Analysis on IMDB Movie Reviews

## Introduction

I explored the IMDB Dataset of 50K Movie Reviews to apply knowledge of word vectors, contextualized embeddings, and sentence transformers.

## Dataset

The dataset has 50,000 balanced data points with two columns: `review` (text) and `sentiment` (positive/negative).

## Exploratory Data Analysis

- No null values.
- Reviews are variable-length texts.
- Sentiments are either positive or negative.

## Text Cleaning

- Removed unwanted characters using regex.
- Tokenized text using NLTK's `word_tokenize` and `sent_tokenize`.

## Label Encoding

- Converted sentiments to binary labels (1 for positive, 0 for negative).

## Vectorization Techniques

### CountVectorizer

- **MLPClassifier:** 89%
- **XGBClassifier:** 85%
- **Logistic Regression:** 89%

### TF-IDF

- **MLPClassifier:** 90%
- **Logistic Regression:** 89%
- **XGBClassifier:** 85%

### Word2Vec

- **MLPClassifier:** 79%
- **Logistic Regression:** 73%
- **XGBClassifier:** 71%

## Sentence Transformers

### all-MiniLM-L6-v2

- Trained on 1 billion pairs.
- Embeddings: 384 dimensions.
- Sequence length: 256.

### bert-base-nli-mean-tokens

- Generates sentence embeddings by averaging token embeddings.

### multi-qa-mpnet-base-dot-v1

- Used for semantic search.
- Embeddings: 768 dimensions.
- Sequence length: 512.

### roberta-large-nli-stsb-mean-tokens

- Fine-tuned on NLI and STSB.
- Mean pooling of token embeddings.

## Conclusion

After extensive experimentation, the best combinations for sentiment analysis on the IMDB dataset are:

1. **TF-IDF with Logistic Regression**
2. **TF-IDF with MLPClassifier**
3. **Sentence Transformer (multi-qa-mpnet-base-dot-v1) with RNN**

I conducted comprehensive experiments to determine the optimal solution for the IMDB 50K dataset.

