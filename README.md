Emotion Detection using Bidirectional LSTM

Overview

This project implements an Emotion Detection system using a Bidirectional LSTM (BiLSTM) model trained on a Kaggle dataset. The model classifies text into six different emotions with high accuracy.

Dataset

The dataset is sourced from Kaggle.

It consists of labeled text samples categorized into six emotions:

Sadness

Joy

Love

Anger

Fear

Surprise

Model Architecture

The model follows these key components:

Embedding Layer: Converts words into dense vector representations.

Bidirectional LSTM Layer: Captures dependencies in both forward and backward directions.

Dense Layer: Outputs a probability distribution over the six emotion classes using softmax activation.

Data Preprocessing

Text Cleaning: Lowercasing, removing HTML tags, punctuation, stopwords, and stemming.

Tokenization & Padding: Converting text to numerical sequences and padding to a fixed length of 50.

Model Training

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Batch Size: 32

Epochs: 5

Performance Metrics

Model Name

Accuracy

F1 Score (Macro)

Recall (Macro)

Precision (Macro)

F1 Score (Micro)

Recall (Micro)

Precision (Micro)

Bidirectional LSTM

0.882

0.834

0.827

0.843

0.882

0.882

0.882

How to Use

Clone the Repository

git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection

Install Dependencies

pip install -r requirements.txt

Run the Model

python train.py

Make Predictions

python predict.py --text "I am very happy today!"

Future Improvements

Implement transformer-based models (BERT, RoBERTa) for better performance.

Deploy the model using Flask or FastAPI.

Develop a real-time emotion detection web app.
