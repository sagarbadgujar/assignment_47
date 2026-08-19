import numpy as np

from sklearn.linear_model import LinearRegression

Border = "-" * 40

# ------------------------------------
# Step 1 : Create Dataset
# ------------------------------------

X = np.array([
    [1, 7],
    [2, 6],
    [3, 7],
    [4, 6],
    [5, 8]
])

Y = np.array([50, 55, 60, 65, 70])

print(Border)
print("Multiple Linear Regression")
print(Border)

print("Input Features:")
print(X)

print("Target:")
print(Y)

# ------------------------------------
# Step 2 : Create Model
# ------------------------------------

Model = LinearRegression()

# ------------------------------------
# Step 3 : Train Model
# ------------------------------------

Model.fit(X, Y)

# ------------------------------------
# Step 4 : Display Coefficients
# ------------------------------------

print(Border)

print("Coefficient of StudyHours :", Model.coef_[0])

print("Coefficient of SleepHours :", Model.coef_[1])

# ------------------------------------
# Step 5 : Display Intercept
# ------------------------------------

print("Intercept :", Model.intercept_)

print(Border)