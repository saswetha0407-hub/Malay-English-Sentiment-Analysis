# Malay-English Sentiment Analysis

## 1. Project Overview

This project is developed for Assignment 3B: Model Implementation (Methodology) for the Natural Language Processing (NLP) course.

The objective of this project is to classify Malay-English tweets into three sentiment categories: Positive, Negative, and Neutral using supervised machine learning techniques.

## 2. Dataset

The dataset used in this project is the Malay-English Sentiment Dataset (MESocSentiment).

Original dataset source:
https://github.com/afifahms/MESocSentiment

The preprocessed dataset contains 599 records.

| Sentiment | Number of Records |
|---|---:|
| Positive | 104 |
| Negative | 16 |
| Neutral | 479 |
| Total | 599 |

The Cleaned_Tweets column is used as the input feature, while the Sentiment (All) column is used as the target variable.

## 3. Methodology

This project applies supervised text classification using two machine learning algorithms:

- Multinomial Naive Bayes (Baseline Model)
- Logistic Regression (Main Model)

Logistic Regression uses balanced class weights to address the imbalanced sentiment distribution.

## 4. NLP Pipeline

Dataset Collection
-> Text Preprocessing
-> Train-Test Split
-> TF-IDF Feature Representation
-> Model Training
-> Sentiment Prediction
-> Model Evaluation

## 5. Feature Representation

Term Frequency-Inverse Document Frequency (TF-IDF) is used to convert textual data into numerical features.

Parameters:

- Maximum features: 5000
- N-gram range: (1, 2)
- Lowercase: True
- Total extracted features: 4075

The TF-IDF vectorizer is fitted only on the training data to prevent data leakage.

## 6. Experimental Setup

The dataset is divided using a stratified 80/20 train-test split.

- Training samples: 479
- Testing samples: 120
- Random state: 42

Model parameters:

Multinomial Naive Bayes:
- Alpha: 1.0

Logistic Regression:
- Maximum iterations: 1000
- Class weight: balanced
- Random state: 42
- C: 1.0

The evaluation metrics include accuracy, macro precision, macro recall, macro F1-score, and confusion matrices.

## 7. Model Performance

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-score |
|---|---:|---:|---:|---:|
| Naive Bayes | 84.17% | 0.6116 | 0.4127 | 0.4315 |
| Logistic Regression | 86.67% | 0.8434 | 0.6632 | 0.6963 |

Logistic Regression achieved better performance than the Naive Bayes baseline on the test dataset and is used for the interactive NLP prototype.

## 8. Tools and Libraries

- Google Colab
- Python 3.13.18
- Pandas 2.2.3
- NumPy 2.4.6
- Scikit-learn 1.6.1
- Matplotlib 3.10.0

## 9. How to Run the Project

1. Download the Jupyter Notebook from this repository.
2. Open the notebook using Google Colab.
3. Upload the preprocessed CSV dataset from Assignment 3A.
4. Run all code cells in sequence.
5. Enter a Malay-English sentence when prompted.
6. View the predicted sentiment and model confidence.

## 10. Working NLP Prototype

The interactive prototype allows users to enter Malay-English sentences and receive a predicted sentiment.

Actual example:

Input: Saya sangat happy dengan service ini

Predicted Sentiment: NEUTRAL

Model Confidence: 50.79%

Although the expected sentiment is positive, the model predicts neutral. This demonstrates a limitation of the current model.

## 11. Limitations

- The dataset is relatively small.
- The sentiment categories are imbalanced.
- Only three negative samples are available in the testing set.
- TF-IDF has limited ability to capture contextual meaning.
- Malay-English code-switching creates additional classification challenges.
- The interactive prototype does not repeat all original text preprocessing steps.

## 12. Conclusion

The project successfully implements a functional NLP sentiment classification prototype. Logistic Regression achieved an accuracy of 86.67% and a macro F1-score of 0.6963, outperforming the Naive Bayes baseline in this experiment.

Future improvements may include collecting more balanced training data, applying consistent input preprocessing, and exploring contextual word embeddings.
