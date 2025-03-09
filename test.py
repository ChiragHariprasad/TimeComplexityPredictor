import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model
model = joblib.load("complexity_model(best).pkl")

# Load the dataset for validation
data = pd.read_csv("complexity_dataset.csv")

# Drop non-numeric columns
X = data.drop(columns=['complexity_label', 'code'])
y = data['complexity_label']

# Convert categorical values ("No", "Yes") into numerical values
X = X.map(lambda x: 1 if x == "Yes" else (0 if x == "No" else x))

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y_encoded, cv=kf, scoring='accuracy')

# Print validation results
print(f"K-Fold Accuracy Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Standard Deviation: {scores.std():.4f}")
