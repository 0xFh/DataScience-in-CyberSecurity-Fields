import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# Load user access log data
data = pd.read_csv('user_access_logs.csv')

# Calculate z-score for each feature
data['feature1_zscore'] = (data['feature1'] - data['feature1'].mean()) / data['feature1'].std()
data['feature2_zscore'] = (data['feature2'] - data['feature2'].mean()) / data['feature2'].std()

# Calculate entropy of each feature
data['feature1_entropy'] = -data['feature1'].value_counts(normalize=True) * np.log2(data['feature1'].value_counts(normalize=True))
data['feature2_entropy'] = -data['feature2'].value_counts(normalize=True) * np.log2(data['feature2'].value_counts(normalize=True))

# Select relevant features and target variable
X = data[['feature1_zscore', 'feature2_entropy']]
y = data['access_granted']

# Train decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Make predictions on test set
y_pred = model.predict(X)

# Evaluate performance using confusion matrix
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)