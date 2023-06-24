import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load URL and webpage features
data = pd.read_csv('url_features.csv')

# Select relevant features and target variable
X = data['url']
y = data['target']

# Apply TF-IDF to convert text data into numerical vectors
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate performance using accuracy and confusion matrix
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"Confusion Matrix:\n{conf_mat}")