from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return {
        "accuracy": accuracy,
        "classification_report": classification,
        "roc_auc": roc_auc
    }
