import os
import pickle
from sklearn.metrics import accuracy_score

import sys
sys.path.append(r"D:\PIHC\commodity-classification\src")
from modules.utils import load_data

def test_model(model_path, test_data_path):
    # Load test data
    df_test = load_data(test_data_path)
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    
    # Load models
    with open(os.path.join(model_path, 'random_forest.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    with open(os.path.join(model_path, 'gradient_boosting.pkl'), 'rb') as f:
        gb_model = pickle.load(f)
    
    # Evaluasi model
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_pred):.4f}")

if __name__ == "__main__":
    model_dir = "models"
    test_file = r".\data\training_data\test.csv"
    test_model(model_dir, test_file)