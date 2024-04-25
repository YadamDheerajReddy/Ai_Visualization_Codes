import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and preprocess the Iris dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and compile a single-layer neural network
model_single_layer = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,)),
    layers.Dense(3, activation='softmax')
])
model_single_layer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_single_layer.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Evaluate the single-layer model
single_layer_predictions = model_single_layer.predict(X_test)
single_layer_accuracy = accuracy_score(y_test, np.argmax(single_layer_predictions, axis=1))
print(f"\nSingle-layer Neural Network - Accuracy: {single_layer_accuracy:.2f}")

# Define and compile a multi-layer neural network
model_multi_layer = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model_multi_layer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_multi_layer.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Evaluate the multi-layer model
multi_layer_predictions = model_multi_layer.predict(X_test)
multi_layer_accuracy = accuracy_score(y_test, np.argmax(multi_layer_predictions, axis=1))
print(f"\nMulti-layer Neural Network - Accuracy: {multi_layer_accuracy:.2f}")
