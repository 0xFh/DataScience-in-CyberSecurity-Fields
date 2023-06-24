import numpy as np

# Load data
data = load_data()

# Calculate mean and standard deviation
mu = np.mean(data)
sigma = np.std(data)

# Calculate Z-score for each data point
z_scores = (data - mu) / sigma