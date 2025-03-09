import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 🔹 Load the dataset (Ensure the filename is correct)
csv_file = "complexity_dataset.csv"
data = pd.read_csv(csv_file)

# 🔹 Display column names (for debugging)
print("Columns in dataset:", data.columns)

# 🔹 Drop 'code' since it's non-numeric, and extract labels
X = data.drop(columns=['complexity_label', 'code'])  # Features
y = data['complexity_label']  # Target variable

# 🔹 Replace 'No' with 0 for numerical consistency
X.replace('No', 0, inplace=True)

# 🔹 Convert all columns to numeric (if necessary)
X = X.apply(pd.to_numeric, errors='coerce')

# 🔹 Handle missing values (if any) using mean imputation
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 🔹 Encode the target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 🔹 Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 🔹 Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 🔹 Make predictions
y_pred = model.predict(X_test_scaled)

# 🔹 Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")

# 🔹 Generate classification report
labels_in_test = np.unique(y_test)
target_names = encoder.inverse_transform(labels_in_test)
print("Classification Report:\n", classification_report(y_test, y_pred, labels=labels_in_test, target_names=target_names))


joblib.dump(model, 'complexity_model(best).pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

