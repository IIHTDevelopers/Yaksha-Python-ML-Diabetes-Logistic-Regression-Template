import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Function 1: Load and preprocess data
def load_and_preprocess(filename='diabetes_data.csv'):
    """
    TODO: Implement this function to load data from the CSV file and preprocess it.
    
    Steps to implement:
    1. Load the diabetes data from the CSV file using pandas
    2. Extract features (X) from columns: 'Age', 'BMI', 'BloodPressure'
    3. Extract target variable (y) from column: 'Diabetic'
    4. Split the data into training and testing sets (80% train, 20% test)
    5. Return X_train, X_test, y_train, y_test
    
    Parameters:
    filename (str): Path to the CSV file containing diabetes data
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test) - Split datasets for training and testing
    Dummy data is provided you will need to change it accordingly 
    """
   
    X_train = np.array([[0, 0, 0]])
    X_test = np.array([[0, 0, 0]])
    y_train = np.array([0])
    y_test = np.array([0])
    return X_train, X_test, y_train, y_test

# Function 2: Train logistic regression model
def train_model(X_train, y_train):
    """
    TODO: Implement this function to train a logistic regression model.
    
    Steps to implement:
    1. Initialize a LogisticRegression model
    2. Fit the model using the training data (X_train, y_train)
    3. Return the trained model
    
    Parameters:
    X_train (DataFrame): Training features
    y_train (Series): Training target variable
    
    Returns:
    LogisticRegression: Trained logistic regression model
    """
   
    return None

# Function 3: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    TODO: Implement this function to evaluate the trained model.
    
    Steps to implement:
    1. Use the model to predict on X_test
    2. Calculate and print accuracy score
    3. Generate and print classification report
    4. Format the output to include "Evaluation Results:", "Accuracy:", and "Classification Report:"
    
    Parameters:
    model (LogisticRegression): Trained logistic regression model
    X_test (DataFrame): Testing features
    y_test (Series): Testing target variable
    
    Returns:
    None: This function prints evaluation metrics but doesn't return anything
    """
    print("Evaluation Results:")
    print("Accuracy: 0.00")

# Function 4: Predict for a new patient
def predict_new(model, age, bmi, bp):
    """
    TODO: Implement this function to make predictions for a new patient.
    
    Steps to implement:
    1. Create a DataFrame with a single row containing the patient's data
    2. Use the model to predict whether the patient is diabetic
    3. Print the prediction result in a formatted string
    
    Parameters:
    model (LogisticRegression): Trained logistic regression model
    age (int/float): Patient's age
    bmi (float): Patient's Body Mass Index
    bp (int/float): Patient's Blood Pressure
    
    Returns:
    None: This function prints the prediction but doesn't return anything
    dummy data provided you will need to change it accordingly 
    """
  
    print(f"Patient with Age={age}, BMI={bmi}, and BP={bp} is not diabetic.\n")

# Function 5: Full workflow execution
def run_pipeline():
    """
    TODO: Implement this function to run the full diabetes prediction pipeline.
    
    Steps to implement:
    1. Load and preprocess the data
    2. Train the model
    3. Evaluate the model
    4. Make predictions for sample patients
    
    Parameters:
    None
    
    Returns:
    None
    """
    # Placeholder implementation
    X_train, X_test, y_train, y_test = load_and_preprocess()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    predict_new(model, 45, 32.5, 88)
    predict_new(model, 25, 22.1, 72)

# Run the full pipeline
if __name__ == '__main__':
    run_pipeline()
