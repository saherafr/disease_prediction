import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Replace zeros with NaN in specific columns
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[columns_with_zeros] = data[columns_with_zeros].replace(0, pd.NA)

    # Fill NaN values with column mean
    data.fillna(data.mean(), inplace=True)

    # Separate features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
