# Sentiment Analysis with NLP using TF-IDF and Logistic Regression
# This file is structured like a Jupyter Notebook
# You can copy it into a .ipynb file for submission

# =====================================
# Cell 1: Import Required Libraries
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# =====================================
# Cell 2: Load Dataset (Sample Reviews Dataset)
# =====================================
# You may replace this with your own dataset (CSV file)

# Sample dataset

data = {
    'review': [
        'This product is amazing and works perfectly',
        'Very bad quality, totally disappointed',
        'I love this phone, great battery life',
        'Worst experience ever, do not buy',
        'Excellent service and fast delivery',
        'Not worth the money',
        'Highly satisfied with the purchase',
        'Terrible product, stopped working',
        'Good quality and reasonable price',
        'Very poor performance'
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

df.head()

# =====================================
# Cell 3: Exploratory Data Analysis
# =====================================
print("Dataset Shape:", df.shape)
print("\nSentiment Distribution:\n", df['sentiment'].value_counts())

# =====================================
# Cell 4: Text Preprocessing
# =====================================

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    words = text.split()
    
    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Apply preprocessing
df['clean_review'] = df['review'].apply(preprocess_text)

df[['review', 'clean_review']].head()

# =====================================
# Cell 5: Feature Extraction using TF-IDF
# =====================================

X = df['clean_review']
y = df['sentiment']

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(X)

print("TF-IDF Shape:", X_tfidf.shape)

# =====================================
# Cell 6: Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# =====================================
# Cell 7: Train Logistic Regression Model
# =====================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =====================================
# Cell 8: Prediction
# =====================================

y_pred = model.predict(X_test)

# =====================================
# Cell 9: Model Evaluation
# =====================================

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =====================================
# Cell 10: Confusion Matrix Visualization
# =====================================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0,1], ['Negative','Positive'])
plt.yticks([0,1], ['Negative','Positive'])
plt.show()

# =====================================
# Cell 11: Testing with New Reviews
# =====================================

def predict_sentiment(text):
    clean_text = preprocess_text(text)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"

# Example tests
print(predict_sentiment("The product is very good and useful"))
print(predict_sentiment("Waste of money and very bad quality"))

# =====================================
# Cell 12: Analysis and Observations
# =====================================
"""
Analysis:

1. Text data was cleaned using lowercase conversion, stopword removal, and stemming.
2. TF-IDF vectorization was used to convert text into numerical features.
3. Logistic Regression was trained for binary sentiment classification.
4. The model was evaluated using accuracy, confusion matrix, and classification report.
5. The system can predict sentiment for new unseen reviews.

Limitations:
- Dataset is small, which may affect performance.
- Larger datasets will improve accuracy.

Improvements:
- Use bigger datasets (Amazon, IMDB reviews)
- Apply lemmatization instead of stemming
- Tune hyperparameters
- Try advanced models (SVM, LSTM, BERT)

Conclusion:
TF-IDF with Logistic Regression provides an effective baseline for sentiment analysis.
"""
