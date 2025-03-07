import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import joblib

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
df = pd.read_csv("cleaned_code_complexity_dataset.csv")

# Handle missing values
df.fillna(0, inplace=True)

# Extract features (X) and labels (y)
X = df[["Loops", "Recursion", "Function_Calls", "If_Conditions"]]
y = df["Complexity"]

# Add polynomial and interaction features
X["Loops_squared"] = X["Loops"] ** 2
X["Function_Calls_squared"] = X["Function_Calls"] ** 2
X["Loops_x_Functions"] = X["Loops"] * X["Function_Calls"]
X["If_x_Recursion"] = X["If_Conditions"] * X["Recursion"]
X["Recursion_squared"] = X["Recursion"] ** 2
X["If_Conditions_squared"] = X["If_Conditions"] ** 2
X["Total_Operations"] = X["Loops"] + X["Recursion"] + X["Function_Calls"] + X["If_Conditions"]

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Handle class imbalance by grouping rare complexity classes
# Define complexity groups
complexity_groups = {
    'Constant': ['O(1)'],
    'Logarithmic': ['O(log n)', 'O(log log n)', 'O(n log log n)'],
    'Linear': ['O(n)', 'O(n + k)', 'O(n+m)', 'O(V + E)'],
    'Linearithmic': ['O(n log n)', 'O(n log k)', 'O(E log E)'],
    'Quadratic': ['O(n^2)', 'O(V^2)'],
    'Cubic': ['O(n^3)', 'O(V^3)'],
    'Polynomial': ['O(n^2 * k)', 'O(m*n)', 'O(n*m)', 'O(n * k)', 'O(n*W)', 'O(n*amount)', 'O(n*sum)', 'O(V * E)'],
    'Exponential': ['O(2^n)', 'O(k^n)', 'O(n^2 * 2^n)', 'O(n!)']
}

# Create reverse mapping
complexity_to_group = {}
for group, complexities in complexity_groups.items():
    for complexity in complexities:
        complexity_to_group[complexity] = group

# Apply grouping
y_grouped = y.map(lambda x: complexity_to_group.get(x, x))
print("Original class count:", len(y.unique()))
print("Grouped class count:", len(y_grouped.unique()))
print("\nComplexity Group Mapping:")
for group, complexities in complexity_groups.items():
    print(f"  {group}: {', '.join(complexities)}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_grouped)

# Print grouped class mapping for reference
class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
print("\nClass Mapping (after grouping):")
for idx, label in class_mapping.items():
    print(f"  {idx}: {label}")

# Check class distribution before split
print("\nClass Distribution After Grouping:", Counter(y_encoded))

# Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
all_models = {}

# Run cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y_encoded)):
    print(f"\n--- Fold {fold+1} ---")
    X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    # Apply SMOTE for oversampling minority classes
    try:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"Class distribution after SMOTE: {Counter(y_train_balanced)}")
    except ValueError as e:
        print(f"SMOTE failed: {e}")
        print("Applying SMOTE with k_neighbors=1")
        try:
            smote = SMOTE(k_neighbors=1, random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"Class distribution after SMOTE with k_neighbors=1: {Counter(y_train_balanced)}")
        except ValueError as e:
            print(f"SMOTE still failed: {e}")
            print("Falling back to original data")
            X_train_balanced, y_train_balanced = X_train, y_train

    # Ensure all classes are present in the training data
    unique_classes = np.unique(y_train_balanced)
    expected_classes = np.arange(len(class_mapping))   # Assuming class_mapping is defined earlier

    if not np.array_equal(np.sort(unique_classes), np.sort(expected_classes)):
        print("Warning: Not all classes are present in the training data after SMOTE.")
        # Optionally, you can manually add missing classes with minimal samples
        missing_classes = np.setdiff1d(expected_classes, unique_classes)
        for cls in missing_classes:
            # Use .iloc to access the first row of the DataFrame
            X_train_balanced = np.vstack([X_train_balanced, X_train_balanced.iloc[0].values])  # Add a dummy sample
            y_train_balanced = np.append(y_train_balanced, cls)

    # Define models
    models = {
        'XGBoost': xgb.XGBClassifier(
            objective='multi:softmax',
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight='balanced',
            random_state=42
        )
    }
    
    best_accuracy = 0
    best_model_name = None
    best_predictions = None
    best_model_obj = None
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_balanced, y_train_balanced)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_predictions = y_pred
            best_model_obj = model
    
    # Save fold accuracy and best model
    fold_accuracies.append(best_accuracy)
    all_models[fold] = {
        'name': best_model_name,
        'model': best_model_obj,
        'accuracy': best_accuracy
    }
    
    # Only print detailed metrics for the best model in this fold
    print(f"\nDetailed metrics for {best_model_name} (best model in fold {fold+1}):")
    print(classification_report(y_test, best_predictions, zero_division=0))
    
    # Generate confusion matrix for this fold
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[class_mapping[i] for i in range(len(class_mapping))],
                yticklabels=[class_mapping[i] for i in range(len(class_mapping))])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Fold {fold+1} - Confusion Matrix - {best_model_name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_fold_{fold+1}.png')
    plt.close()
    
    # Feature importance for this fold
    if best_model_name == 'XGBoost':
        feature_importance = best_model_obj.feature_importances_
        feature_names = X.columns
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=False)
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Fold {fold+1} - XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig(f'feature_importance_fold_{fold+1}.png')
        plt.close()
    elif best_model_name == 'RandomForest':
        feature_importance = best_model_obj.feature_importances_
        feature_names = X.columns
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=False)
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Fold {fold+1} - Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(f'feature_importance_fold_{fold+1}.png')
        plt.close()

print(f"\nAverage Cross-Validation Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")

# Find the best model across all folds
best_fold = max(all_models.keys(), key=lambda k: all_models[k]['accuracy'])
best_overall_model = all_models[best_fold]['model']
best_overall_model_name = all_models[best_fold]['name']
print(f"\nBest overall model: {best_overall_model_name} from fold {best_fold+1} with accuracy {all_models[best_fold]['accuracy']*100:.2f}%")

# Define a prediction function using the best model
def predict_complexity(loops, recursion, function_calls, if_conditions):
    # Generate the additional features
    features = {
        "Loops": loops,
        "Recursion": recursion,
        "Function_Calls": function_calls,
        "If_Conditions": if_conditions,
        "Loops_squared": loops ** 2,
        "Function_Calls_squared": function_calls ** 2,
        "Loops_x_Functions": loops * function_calls,
        "If_x_Recursion": if_conditions * recursion,
        "Recursion_squared": recursion ** 2,
        "If_Conditions_squared": if_conditions ** 2,
        "Total_Operations": loops + recursion + function_calls + if_conditions
    }
    
    # Convert to DataFrame and scale
    features_df = pd.DataFrame([features])
    features_scaled = pd.DataFrame(scaler.transform(features_df), columns=features_df.columns)
    
    # Make prediction
    prediction = best_overall_model.predict(features_scaled)[0]
    return class_mapping[prediction]

# Example Predictions
print("\nExample Predictions:")
examples = [
    (1, 0, 1, 0),
    (2, 0, 1, 0),
    (3, 1, 2, 1),
    (0, 1, 0, 2),
    (5, 0, 3, 1)
]

for example in examples:
    loops, recursion, function_calls, if_conditions = example
    complexity = predict_complexity(loops, recursion, function_calls, if_conditions)
    print(f"Code with {loops} loops, {recursion} recursion, {function_calls} function calls, {if_conditions} if conditions → {complexity}")

# Save the final model
joblib.dump(best_overall_model, 'code_complexity_predictor_model.joblib')
joblib.dump(scaler, 'code_complexity_scaler.joblib')
joblib.dump(label_encoder, 'code_complexity_label_encoder.joblib')

print("\nModel saved as 'code_complexity_predictor_model.joblib'")
