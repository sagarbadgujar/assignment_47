import numpy as np
from sklearn.linear_model import LinearRegression

# Input data - Study Hours
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Output data - Marks
Y = np.array([50, 55, 60, 65, 70])

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# Display coefficient and intercept
print("Coefficient :", model.coef_[0])
print("Intercept   :", model.intercept_)