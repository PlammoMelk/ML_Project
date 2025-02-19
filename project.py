import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import joblib
import os
import numpy as np

def load_arff_to_dataframe(file_path):
    """Load an ARFF file into a pandas DataFrame."""
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Decode byte strings if necessary
    df = df.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    return df

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train the given model, evaluate performance, and print metrics."""
    print(f"\n========== Training {model_name} ==========")
    
    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate various performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=["win", "loss", "draw"])

    # Print model evaluation metrics
    print(f"\n========== {model_name} Performance ==========")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Confusion Matrix with Labels
    print("\n========== Confusion Matrix ==========")
    print("        Predicted: Win  Predicted: Loss  Predicted: Draw")
    print("Actual Win  ", conf_matrix[0])
    print("Actual Loss ", conf_matrix[1])
    print("Actual Draw ", conf_matrix[2])

    # Classification Report
    print("\n========== Classification Report ==========")
    print(class_report)

    # Feature Importance (only for tree-based models)
    if hasattr(model, "feature_importances_"):
        print("\n========== Feature Importance ==========")
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]  # Sort by importance
        for i in range(10):  # Print top 10 most important features
            print(f"{X_train.columns[sorted_indices[i]]}: {feature_importances[sorted_indices[i]]:.4f}")

    # Prediction Distribution (See how often each class is predicted)
    unique, counts = np.unique(y_pred, return_counts=True)
    prediction_distribution = dict(zip(["win", "loss", "draw"], counts))
    print("\n========== Prediction Distribution ==========")
    print(f"Predicted Wins: {prediction_distribution.get('win', 0)}")
    print(f"Predicted Losses: {prediction_distribution.get('loss', 0)}")
    print(f"Predicted Draws: {prediction_distribution.get('draw', 0)}")

    # Save the trained model
    model_filename = f"{model_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, model_filename)
    print(f"\nModel saved as {model_filename}")

    return model, model_filename

def main():
    """Main function to load data, train models, and test predictions."""
    file_path = r"C:\\Users\\Owen\Documents\\ML_Project\\connect_4_data.arff"  # Update with your actual file path
    
    # Load dataset
    df = load_arff_to_dataframe(file_path)
    
    print("First 5 rows of the dataset:")
    print(df.head())

    # Prepare features and labels
    X = df.drop(columns=["class"])  # Remove the target column
    y = df["class"].astype("category").cat.codes  # Convert categorical labels to numerical

    # Split data into training (80%) and testing (20%) - Same split for all models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")

    # Train Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    train_and_evaluate_model(dt_model, X_train, X_test, y_train, y_test, "Decision Tree")

    # Standardize features for MLP Classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train MLP Classifier (Neural Network)
    mlp_model = MLPClassifier(random_state=42, max_iter=500)
    train_and_evaluate_model(mlp_model, X_train_scaled, X_test_scaled, y_train, y_test, "MLP Classifier")

if __name__ == "__main__":
    main()
