import numpy as np
import matplotlib.pyplot as plt

# Create circle data
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)]).T

# Create square data
square = np.array([
    [x, y] for x in np.linspace(-1, 1, 50)
    for y in [-1, 1]
] + [
    [x, y] for y in np.linspace(-1, 1, 50)
    for x in [-1, 1]
])

# Add noise
circle += np.random.normal(0, 0.1, circle.shape)
square += np.random.normal(0, 0.1, square.shape)

data = np.vstack((circle, square))

# Initialize 2 neurons
weights = np.random.randn(2, 2)

learning_rate = 0.05
epochs = 30

# Training
for epoch in range(epochs):
    for point in data:
        distances = np.linalg.norm(weights - point, axis=1)
        winner = np.argmin(distances)

        weights[winner] += learning_rate * (point - weights[winner])

# Plot
plt.scatter(circle[:,0], circle[:,1], label="Circle Data")
plt.scatter(square[:,0], square[:,1], label="Square Data")
plt.scatter(weights[:,0], weights[:,1], color='red', s=100, label="Neurons")

plt.legend()
plt.title("Shape-Based Competitive Learning")
plt.show()

print("Final neuron positions:")
print(weights)
