import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create DataFrame
data = pd.DataFrame({
    'Weather': ['Sunny', 'Rainy', 'Sunny', 'Sunny'],
    'Wind': ['Mild', 'Mild', 'High', 'Mild'],
    'Temp': ['Moderate', 'Mild', 'Moderate', 'Mild'],
    'Go': ['Yes', 'No', 'Yes', 'Yes']
})

# Display DataFrame columns
print(data.columns)

# Encode labels
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

# Split dataset into features and target
X = data.iloc[:, :3]
y = data.iloc[:, 3]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
