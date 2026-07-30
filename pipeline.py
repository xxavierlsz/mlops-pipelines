import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn

def load_data(path="data/iris.csv"):
    # For demonstration, using a simple dataset
    if not os.path.exists(path):
        # Create a dummy iris.csv if it doesn\'t exist
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    return pd.read_csv(path)

def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "random_forest_model")
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy

if __name__ == "__main__":
    # Ensure MLflow tracking is set up
    mlflow.set_tracking_uri("file:///tmp/mlruns") # Local tracking
    mlflow.set_experiment("MLOps_Classification_Pipeline")

    # Load data
    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train, n_estimators=150, max_depth=12)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model locally (for deployment simulation)
    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")
