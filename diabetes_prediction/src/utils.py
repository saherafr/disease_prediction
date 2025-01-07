def print_evaluation_results(results):
    print("Accuracy:", results['accuracy'])
    print("Classification Report:\n", results['classification_report'])
    print("ROC-AUC Score:", results['roc_auc'])
import joblib

def load_model(file_path):
    return joblib.load(file_path)
