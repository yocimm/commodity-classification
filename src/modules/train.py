import os
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import sys
sys.path.append(r"D:\PIHC\commodity-classification\src")
from modules.utils import load_data

def train_model(train_data_path, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Load train data
    df_train = load_data(train_data_path)
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Train Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    
    # Save models
    with open(os.path.join(model_path, 'random_forest.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    with open(os.path.join(model_path, 'gradient_boosting.pkl'), 'wb') as f:
        pickle.dump(gb_model, f)
    
    print("Training selesai. Model disimpan di folder model.")

if __name__ == "__main__":
    train_file = r".\data\training_data\train.csv"
    model_dir = "models"
    train_model(train_file, model_dir)