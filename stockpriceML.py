#pip install yfinance pandas scikit-learn matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# Download stock data (Example: Apple - AAPL)
stock_symbol = "AAPL"
df = yf.download(stock_symbol, start="2015-01-01", end="2024-01-01")

# Show first few rows
print(df.head())

# Select 'Close' price as target
df = df[['Close']]

# Create features and target
df['Prediction'] = df['Close'].shift(-30)  # Predict the next 30 days

# Drop last 30 rows (NaN values after shift)
df.dropna(inplace=True)

# Split into features (X) and target (y)
X = np.array(df[['Close']])
y = np.array(df['Prediction'])

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Support Vector Regression model
model = SVR(kernel='rbf', C=1000, gamma=0.1)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)


plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Prices")
plt.plot(predictions, label="Predicted Prices")
plt.legend()
plt.title(f"Stock Price Prediction for {stock_symbol}")
plt.show()
