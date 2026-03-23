import numpy as np

# Step function (activation)
def step_function(x):
    return 1 if x >= 0 else 0

# Training data (AND gate)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.zeros(2)
bias = 0
learning_rate = 0.1

# Training
epochs = 10

for epoch in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step_function(linear_output)

        error = y[i] - prediction

        # Update weights and bias
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Testing
print("Final Weights:", weights)
print("Final Bias:", bias)

print("\nPredictions:")
for i in range(len(X)):
    result = step_function(np.dot(X[i], weights) + bias)
    print(X[i], "->", result)
