import numpy as np
from hmmlearn import hmm

# Step 1: Define Model Parameters
n_states = 2  # Number of hidden states: Rainy and Sunny

# Transition matrix (A): Probability of transitioning from one state to another
trans_matrix = np.array([
    [0.7, 0.3],  # Transition from Rainy to Rainy and Sunny
    [0.4, 0.6]   # Transition from Sunny to Rainy and Sunny
])

# Emission matrix (B): Probability of observing an emission given the current state
emission_matrix = np.array([
    [0.1, 0.4, 0.5],  # Emissions when Rainy: Umbrella, Jacket, T-shirt
    [0.6, 0.3, 0.1]   # Emissions when Sunny: Umbrella, Jacket, T-shirt
])

# Initial state probabilities (π): Probability distribution of starting in each state
initial_probs = np.array([0.6, 0.4])

# Step 2: Create HMM Model
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = initial_probs
model.transmat_ = trans_matrix
model.emissionprob_ = emission_matrix
model.n_iter = 100

# Step 3: Generate Training Data (for simplicity, this is hypothetical)
# Observations: 0 - Umbrella, 1 - Jacket, 2 - T-shirt
train_data = np.array([[0, 1, 2, 0, 1, 2, 0, 2, 1]])

# Reshape the array if needed
train_data = train_data.reshape(-1, 1)

# Step 4: Fit the Model
model.fit(train_data)

# Step 5: Predict States for a New Sequence
new_data = np.array([[0, 2, 1]])  # Umbrella, T-shirt, Jacket
new_data = new_data.reshape(-1, 1)
predicted_states = model.predict(new_data)

# Map numerical predictions to weather states
weather_states = ['Rainy', 'Sunny']
predicted_states_text = [weather_states[state] for state in predicted_states]

# Display Results
print("Predicted Weather States:", predicted_states_text)
