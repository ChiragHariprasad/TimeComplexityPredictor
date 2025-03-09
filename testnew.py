import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the unseen GitHub dataset
unseen_data = pd.read_csv("github_code_samples.csv")

# Load true labels (assumes a column named 'true_complexity')
true_labels = unseen_data['complexity']

# Drop non-numeric columns
X_unseen = unseen_data.drop(columns=['code', 'true_complexity'])

# Load the trained model
with open("complexity_model(best).pkl", "rb") as f:
    model = pickle.load(f)

# Recreate and fit the scaler using training data statistics
train_data = pd.read_csv("complexity_dataset.csv")  # Load the original training data
X_train = train_data.drop(columns=['code', 'complexity_label'])  # Remove non-numeric columns

scaler = StandardScaler()
scaler.fit(X_train)  # Fit scaler on training data

# Scale the unseen dataset
X_unseen_scaled = scaler.transform(X_unseen)

# Predict on unseen data
y_pred = model.predict(X_unseen_scaled)

# Evaluate accuracy
accuracy = accuracy_score(true_labels, y_pred)
print(f"Accuracy on unseen data: {accuracy:.4f}")

# Generate classification report
print("Classification Report:\n", classification_report(true_labels, y_pred))
