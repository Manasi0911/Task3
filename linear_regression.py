import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv('dataset.csv')
print("Dataset Head:\n", df.head())

# 2. Define X (independent variable) and y (dependent/target variable)
X = df[['Experience']]   # simple linear regression (one feature)
y = df['Salary']

# 3. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)

# 7. Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# 8. Display model coefficients
print("\nModel Coefficients:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])
