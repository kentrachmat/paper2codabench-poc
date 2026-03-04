import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

def predict(input_dir, output_dir):
    # Read the input data
    input_path = Path(input_dir) / 'input.csv'
    data = pd.read_csv(input_path)
    
    # Extract features and target
    features = ['ux', 'uy', 'p', 'νt', 'ps', 'CL', 'CD', 'ρD']
    target = 'ρL'
    
    # For simplicity, we will use a heuristic: assume a linear relationship
    # Generate synthetic training data based on the input data
    # Here, we assume ρL is linearly dependent on the input features
    np.random.seed(42)  # For reproducibility
    num_samples = len(data)
    X_train = np.random.rand(num_samples, len(features))
    y_train = np.dot(X_train, np.random.rand(len(features)))  # Generate synthetic target values
    
    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prepare the input features for prediction
    X_test = data[features].replace('value_', '', regex=True).astype(float)
    
    # Predict ρL
    data[target] = model.predict(X_test)
    
    # Write the predictions to output_dir
    output_path = Path(output_dir) / 'predictions.csv'
    data.to_csv(output_path, index=False)