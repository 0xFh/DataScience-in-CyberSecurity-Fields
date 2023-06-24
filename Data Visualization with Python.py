import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
X, y = load_data()

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Create scatter plot
plt.scatter(X, y)

# Plot linear regression line
plt.plot(X, model.predict(X), color='red')

# Add axis labels and title
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.title('Scatter Plot with Linear Regression Line')
plt.show()