import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load PDF metadata and structural features
data = pd.read_csv('pdf_features.csv')

# Select relevant features and target variable
X = data.drop(['filename', 'target'], axis=1)
y = data['target']

# Apply PCA to reduce dimensionality
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train SVM classifier
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate performance using accuracy and confusion matrix
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"Confusion Matrix:\n{conf_mat}")