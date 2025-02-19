import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_arff_to_dataframe(file_path):
    """Load an ARFF file into a pandas DataFrame."""
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Decode byte strings if necessary
    df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    return df

def train_model(df):
    """Train a Random Forest classifier using the Connect-4 dataset."""
    # Prepare features and labels
    X = df.drop(columns=["class"])  # Remove the target column
    y = df["class"].astype("category").cat.codes  # Convert categorical labels to numerical

    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Save the trained model
    model_filename = "connect4_rf_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

    return model, model_filename

def predict_outcome(model, new_board_state):
    """Predict the game outcome for a given board state."""
    # Convert board state into a DataFrame
    new_board_df = pd.DataFrame([new_board_state], columns=[f"{col}{row}" for col in "abcdefg" for row in range(1, 7)])
    
    # Make prediction
    predicted_class = model.predict(new_board_df)[0]
    
    # Convert numeric prediction back to class labels
    class_mapping = {0: "win", 1: "loss", 2: "draw"}
    return class_mapping[predicted_class]

def main():
    """Main function to load data, train model, and test prediction."""
    file_path = r"C:\\Users\\Owen\Documents\\ML_Project\\connect_4_data.arff"  # Update with your actual file path
    df = load_arff_to_dataframe(file_path)
    
    print("First 5 rows of the dataset:")
    print(df.head())  

    # Train the model
    model, model_filename = train_model(df)

    # Example board state for prediction (Replace with actual board state)
    example_board = [0] * 42  # A board with all empty spaces
    predicted_outcome = predict_outcome(model, example_board)
    
    print(f"Predicted outcome for the given board state: {predicted_outcome}")

if __name__ == "__main__":
    main()
