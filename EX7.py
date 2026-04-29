import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate values
x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y)

plt.title('Visualization of the Sigmoid Function')
plt.xlabel('Input')
plt.ylabel('Output')

plt.grid()
plt.show()
