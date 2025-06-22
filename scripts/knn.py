from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

import os
import joblib
import numpy as np

def get_scene_KNN(all_points, all_labels, scene_token):
    output_dir = f"./model/KNN/{scene_token}"
    model_filename = os.path.join(output_dir, 'knn_model_all.joblib')
    scaler_filename = os.path.join(output_dir, 'scaler_model_all.joblib')
    
    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        print("Model and scaler already exist. Loading...")
        # final_knn_classifier = joblib.load(model_filename)
        # scaler = joblib.load(scaler_filename)
    else:
        print("Model and scaler not found. Training new model...")
        # Split into points and labels
        X_train, X_test, y_train, y_test = train_test_split(all_points, all_labels, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define a range of K values for cross-validation
        k_values = list(range(1, 21))
        best_k = None
        best_accuracy = 0

        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            cross_val_scores = cross_val_score(knn_classifier, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=5))
            average_accuracy = np.mean(cross_val_scores)

            if average_accuracy > best_accuracy:
                best_accuracy = average_accuracy
                best_k = k

        print("Best K value:", best_k)
        print("Best Cross-Validation Accuracy:", best_accuracy)

        # Train the final model with the best K value
        final_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
        final_knn_classifier.fit(X_train_scaled, y_train)

        # Predict on the test set
        y_pred = final_knn_classifier.predict(X_test_scaled)

        # Evaluate the final model
        accuracy = np.mean(y_pred == y_test)
        print("Final Model Accuracy:", accuracy)

        os.makedirs(output_dir, exist_ok=True)

        joblib.dump(final_knn_classifier, model_filename)
        joblib.dump(scaler, scaler_filename)
        print("Model and scaler saved to", output_dir)
