import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import joblib

# Load the breast cancer dataset
def prepare_data():
    # Load breast cancer dataset from sklearn
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    print("Loading and preparing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data()
    
    print("Training model...")
    model = train_model(X_train_scaled, y_train)
    
    # Calculate and print test accuracy
    test_accuracy = model.score(X_test_scaled, y_test)
    print(f"Model test accuracy: {test_accuracy:.4f}")
    
    # Save the model and scaler
    print("Saving model and scaler...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully!")
    
    # Create example patient data
    print("Creating example patient data file...")
    data = load_breast_cancer()
    example_data = pd.DataFrame(data.data[:5], columns=data.feature_names)
    example_data.to_csv('example_patients.csv', index=False)
    print("Example patient data saved as 'example_patients.csv'")

if __name__ == "__main__":
    main() 