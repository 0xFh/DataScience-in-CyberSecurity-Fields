import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Load email data
data = pd.read_csv('email_data.csv')

# Select relevant features and target variable
X = data['text']
y = data['label']

# Convert text data into numerical vectors using CountVectorizer
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate performance using accuracy and confusion matrix
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"Confusion Matrix:\n{conf_mat}")