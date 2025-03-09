import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load("complexity_model(best).pkl")

# Load test dataset
data = pd.read_csv("complexity_dataset.csv")

# Drop non-numeric columns
X = data.drop(columns=['complexity_label', 'code'])
y = data['complexity_label']

# Convert categorical values ("Yes", "No") into numeric
X = X.map(lambda x: 1 if x == "Yes" else (0 if x == "No" else x))

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predict with the model
y_pred = model.predict(X_scaled)

# Generate Classification Report
print("📊 Classification Report:\n", classification_report(y_encoded, y_pred, target_names=encoder.classes_))

# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_encoded, y_pred)

# Plot and Save Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # ✅ Save as image
plt.close()

# Feature Importance
feature_importances = model.feature_importances_
feature_names = X.columns

# Plot and Save Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names, hue = feature_names, palette="viridis")
plt.legend([], [], frameon=False)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Complexity Prediction")
plt.savefig("feature_importance.png")  # ✅ Save as image
plt.close()

print("✅ Confusion Matrix saved as 'confusion_matrix.png'")
print("✅ Feature Importance saved as 'feature_importance.png'")
