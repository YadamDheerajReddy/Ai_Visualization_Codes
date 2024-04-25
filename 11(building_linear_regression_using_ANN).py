import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(0)
X_train = np.random.rand(100, 1)
y_train = 2 * X_train + 1 + 0.1 * np.random.randn(100, 1)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(1,))
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, verbose=0)

# Plot the training loss over epochs
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error Loss')
plt.title('Training Loss')
plt.show()

# Make predictions on new data
X_test = np.array([[0.2], [0.5], [0.8]])
predictions = model.predict(X_test)

# Print the predictions
print("Predicted values:")
for i, prediction in enumerate(predictions):
    print(f"Input: {X_test[i][0]}, Predicted: {prediction[0]}")
