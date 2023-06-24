from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load data
X, y = load_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict class probabilities for test data
probs = model.predict_proba(X_test)

# Calculate overall probability of each class
class_probs = np.sum(probs, axis=0)