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

## Top Algorithms for Vectorization and Word Embeddings 

### CountVectorizer

-CountVectorizer is a straightforward algorithm that converts textual data into numerical vectors by counting the occurrence of each word in the text.

- The MLPClassifier model, when trained on the word vectors generated by the CountVectorizer algorithm, achieved an impressive accuracy score of 90%. In comparison, the Logistic regression model scored 89%, and the XGBClassifier model performed at 85%.

### TF-IDF

-This technique is a statistical measure to evaluate the importance of a word relative to a collection of documents(corpus).

- The MLPClassifier model worked best with the vectors produced by the TF-IDF algorithm and got an outstanding accuracy of 90%. The Logistic regression model and the XGBClassifier models also performed well, achieving 89% and 85% accuracy, respectively.

### Word2Vec-- Skip-gram Model

-The Skip-gram model is a type of neural network model used for learning word embeddings, which are dense vector representations of words in a continuous vector space.
  
- The Skip-gram algorithm of the Word2Vec model could not perform well yet achieved good scores. The MLPClassifier accuracy was 79%, the logistic regression model got 73%, and the XGBClassofier could achieve 71%.

## Sentence Transformers

### all-MiniLM-L6-v2

- This sentence transformer model was trained on 1 billion pairs of input/output. Its embeddings have 384 dimensions. It takes a maximum input of sequence length 256.

### bert-base-nli-mean-tokens

- This model generates sentence embeddings by taking the average of token embeddings.

### multi-qa-mpnet-base-dot-v1

It is widely used for semantic search. Its embeddings have 768 dimensions. It takes a maximum input of sequence length 512.

### roberta-large-nli-stsb-mean-tokens

- The model is fine-tuned on NLI and STSB. It produces a mean pooling of token embeddings.

## Conclusion

After extensive experimentation, the best combinations for sentiment analysis on the IMDB dataset are:

1. **TF-IDF with Logistic Regression**
2. **TF-IDF with MLPClassifier**
3. **Sentence Transformer (multi-qa-mpnet-base-dot-v1) with RNN**

I conducted comprehensive experiments to determine the optimal solution for the IMDB 50K dataset.

