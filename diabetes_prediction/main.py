from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_logistic_regression, train_random_forest, save_model
from src.model_evaluation import evaluate_model
from src.utils import print_evaluation_results
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Step 1: Load and preprocess data
file_path = "data/diabetes.csv"
data = load_data(file_path)
X, y = preprocess_data(data)

# Step 2: Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 3: Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 4: Train models
logistic_model = train_logistic_regression(X_train, y_train)
random_forest_model = train_random_forest(X_train, y_train)

# Step 5: Evaluate models
logistic_results = evaluate_model(logistic_model, X_test, y_test)
random_forest_results = evaluate_model(random_forest_model, X_test, y_test)

# Step 6: Print results
print("Logistic Regression Results:")
print_evaluation_results(logistic_results)

print("\nRandom Forest Results:")
print_evaluation_results(random_forest_results)

# Step 7: Save the best model
save_model(random_forest_model, "models/best_diabetes_model.pkl")
