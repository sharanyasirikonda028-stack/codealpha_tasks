import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("car data.csv")

# Display first rows
print(data.head())

# Check missing values
print(data.isnull().sum())

# Display column names
print(data.columns)

# Convert categorical columns using encoding
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0, 'Diesel':1, 'CNG':2})
data['Selling_type'] = data['Selling_type'].map({'Dealer':0, 'Individual':1})

data['Transmission'] = data['Transmission'].map({'Manual':0, 'Automatic':1})

# Create feature and target variables
X = data[['Present_Price', 'Driven_kms', 'Fuel_Type',
          'Selling_type', 'Transmission', 'Owner']]

y = data['Selling_Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Error checking
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# Visualization
plt.scatter(y_test, predictions)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Car Price Prediction")

plt.show()