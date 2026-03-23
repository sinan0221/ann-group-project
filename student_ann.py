import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Input data (study hours, sleep, attendance)
X = np.array([
    [2, 5, 50],
    [3, 6, 60],
    [4, 5, 65],
    [6, 7, 80],
    [7, 8, 90],
    [8, 7, 85],
    [1, 4, 40],
    [2, 3, 30]
])

# Output (0 = Fail, 1 = Pass)
y = np.array([0, 0, 0, 1, 1, 1, 0, 0])

# Normalize input
X = X / np.max(X, axis=0)

# Build model
model = Sequential()
model.add(Dense(5, input_dim=3, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=100, verbose=0)

# Test prediction
test = np.array([[5, 6, 70]]) / np.max(X, axis=0)
prediction = model.predict(test)

print("Prediction (Pass=1, Fail=0):", prediction)
