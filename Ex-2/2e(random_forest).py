import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
from matplotlib import pyplot as plt

# Load the dataset
data = pd.read_csv(r"heart.csv")

# Select features and target variable
X = data.iloc[:, 0:12].values  # Adjusted to include the first 12 columns
y = data.iloc[:, 13].values   # Target variable, assuming column index 13 is correct

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Predict on the test set
y_pred = rfc.predict(X_test)

# Classification metrics
print("Classification Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)

# Confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

# Visualizing the confusion matrix
plt.figure(figsize=(5, 4))
sn.heatmap(cm, annot=True, fmt='g')  # 'fmt' is added to avoid scientific notation
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()
