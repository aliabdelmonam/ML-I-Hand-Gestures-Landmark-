import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
import sys
import platform
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
from data_preprocessing import read_data, data_preprocess
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import sys







RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
import random
random.seed(RANDOM_STATE)


# Ensure models and visualization directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("visualization", exist_ok=True)

# Read Data
data_path = Path("dataset") / "hand_landmarks_data.csv"
df = read_data(file_path=data_path)
X_train, X_test, y_train, y_test = data_preprocess(df)

# Cross-validation setup
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# Model definitions
models = {
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'params': {
            'max_depth': [8, 12, 18, 30, 50],
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'params': {
            'n_estimators': [150, 300, 600, 1000],
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=5000, n_jobs=-1),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
        }
    },
    'SVM': {
        'model': SVC(random_state=RANDOM_STATE),
        'params': {
            'C': [0.1, 1, 10,100,120,150,170],
            'kernel': ['rbf'],
        }
    },
}


#==========mlflow===============
mlflow.set_tracking_uri("http://localhost:5000")
exp = mlflow.set_experiment("Hand-Gestures Landmarks")
exp_id = exp.experiment_id



# Grid search and MLflow logging

for model_name, model_info in models.items():
    run = None
    try:
        run = mlflow.start_run(run_name=model_name)
        grid = GridSearchCV(model_info['model'], model_info['params'], cv=cv, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Infer and log signature
        model_signature = mlflow.models.infer_signature(X_test, y_pred)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        # Log metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")

        # Print Metrics
        print(f"--- {model_name} Test Dataset Evaluation ---")
        print(f"  Accuracy  = {acc:.4f}")
        print(f"  F1        = {f1:.4f}")
        print(f"  Recall    = {recall:.4f}")
        print(f"  Precision = {precision:.4f}")

        # Log errors if any
        if hasattr(grid, 'cv_results_'):
            mlflow.log_dict(grid.cv_results_, "cv_results.json")

        # Log model with signature using model name as artifact path
        mlflow.sklearn.log_model(best_model, artifact_path=model_name.replace(" ", "_"), signature=model_signature)



        # Visualization: classification report as image
        report = classification_report(y_test, y_pred, output_dict=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        table = ax.table(cellText=[list(report[cls].values()) for cls in report if isinstance(report[cls], dict)],
                         rowLabels=[cls for cls in report if isinstance(report[cls], dict)],
                         colLabels=list(next(iter(report.values())).keys()),
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        plt.title(f"Classification Report: {model_name}")
        vis_path = Path("visualization") / f"{model_name}_report_{run_id}.png"
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Visualization saved: {vis_path}")
        mlflow.log_artifact(str(vis_path))
    except Exception as e:
        print(f"ERROR in training, saving, or visualizing {model_name}: {e}")
    finally:
        if run is not None:
            mlflow.end_run()

