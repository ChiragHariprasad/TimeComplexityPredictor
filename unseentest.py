import pandas as pd
import joblib

# Load the trained model
model = joblib.load("complexity_model(best).pkl")  # Update with actual model filename

# Load the unseen GitHub code samples
github_data = pd.read_csv("github_code_samples.csv")  # Update with correct path

# Ensure required columns exist
if 'loops' not in github_data.columns:
    github_data['loops'] = 0  # Placeholder, replace with real extraction logic
if 'recursion' not in github_data.columns:
    github_data['recursion'] = 0  # Placeholder
if 'function_calls' not in github_data.columns:
    github_data['function_calls'] = 0  # Placeholder
if 'if_conditions' not in github_data.columns:
    github_data['if_conditions'] = 0  # Placeholder

# Select only relevant feature columns
X_unseen = github_data[['loops', 'recursion', 'function_calls', 'if_conditions']]

# Predict complexity
predicted_complexity = model.predict(X_unseen)

# Add predictions to the dataframe
github_data['predicted_complexity'] = predicted_complexity

# Save results
github_data.to_csv("predicted_github_samples.csv", index=False)

print("Predictions saved to predicted_github_samples.csv")
