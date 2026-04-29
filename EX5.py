import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
iris = load_iris()

# Convert to DataFrame
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

print(data.head())

# Features and target
X = data[['sepal length (cm)']]
y = data['sepal width (cm)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse:.2f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='red', marker='*', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=2, label='Predicted')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Linear Regression: Sepal Width vs Sepal Length')

plt.legend()
plt.show()

# Prediction for new value
new_sample = pd.DataFrame([[5]], columns=['sepal length (cm)'])
predicted_width = model.predict(new_sample)

print(f"The predicted sepal width for sepal length {new_sample.values.tolist()} is {predicted_width[0]:.2f} cm")
