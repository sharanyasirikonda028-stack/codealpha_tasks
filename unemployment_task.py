import pandas as pd
import matplotlib.pyplot as plt


# Load Dataset
data = pd.read_csv("Unemployment in India.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Display first 5 rows
print(data.head())

# Dataset information
print(data.info())

# Check missing values
print(data.isnull().sum())

# Display column names
print(data.columns)

# Basic statistics
print(data.describe())

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Remove missing values
data = data.dropna()

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Visualization
plt.figure(figsize=(12,5))

# Plot unemployment rate
monthly_avg = data.groupby('Date')['Estimated Unemployment Rate (%)'].mean()

plt.plot(monthly_avg.index, monthly_avg.values)


plt.xticks(rotation=90)

plt.title("Unemployment Rate in India")
plt.xlabel("Date")
plt.ylabel("Estimated Unemployment Rate (%)")

plt.show()