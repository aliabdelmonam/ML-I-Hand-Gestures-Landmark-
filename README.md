# Hand-Gestures Landmarks MLflow Project

## Overview
This project trains and evaluates multiple classification models for hand gesture recognition using landmark data. It leverages MLflow for experiment tracking, model management, and artifact logging.

## Structure
- `src/`: Contains data preprocessing scripts.
- `dataset/`: Contains the raw data (`hand_landmarks_data.csv`).
- `model_training.py`: Main script for training, grid search, evaluation, and MLflow logging.
- `models/`: Stores trained model files for each run.
- `visualization/`: Stores classification report images for each model/run.

## Workflow
1. **Data Loading & Preprocessing**
   - Loads the dataset and preprocesses it for training/testing.
2. **Model Training & Grid Search**
   - Trains Decision Tree, Random Forest, Logistic Regression, and SVM classifiers.
   - Performs grid search for hyperparameter optimization.
3. **MLflow Tracking**
   - Each model is logged as a separate MLflow run.
   - Parameters, metrics, confusion matrix, and classification report images are logged as artifacts.
   - Models are logged with their names for easy identification.
4. **Visualization**
   - Classification report for each model is saved as an image in the `visualization/` folder and logged to MLflow.
5. **Model Registry**
   - The best performing model (SVM) was registered in the MLflow Model Registry for future deployment and management.

## How to Run
1. Start the MLflow tracking server:
   ```bash
   mlflow server --port 5000
   ```
2. Run the training script:
   ```bash
   python model_training.py
   ```
3. View results and artifacts in the MLflow UI:
   - Open [http://localhost:5000](http://localhost:5000) in your browser.

## Notes
- Each model run is tracked independently in MLflow.
- Model artifacts and visualizations are versioned by run ID.
- The SVM model was registered in the MLflow Model Registry as the best model.

## models 
- Didn't upload it because of the large size
