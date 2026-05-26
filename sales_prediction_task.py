import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("Advertising.csv")

# Display first rows
print(data.head())

# Check missing values
print(data.isnull().sum())

# Display columns
print(data.columns)

# Features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Error checking
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# Visualization
plt.scatter(y_test, predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Sales Prediction")

plt.show()