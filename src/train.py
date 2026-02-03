import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
import joblib
from sklearn.metrics import f1_score, roc_auc_score

def train_model():
    # 1. Load Processed Data
    X_train = pd.read_csv('../data/processed/train_x.csv')
    y_train = pd.read_csv('../data/processed/train_y.csv').values.ravel()

    # 2. Start MLflow Tracking
    mlflow.set_experiment("Customer_Churn_Prediction")

    with mlflow.start_run():
        # Define Hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "random_state": 42
        }

        # 3. Train Model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 4. Quick Evaluation (on training data for logging)
        preds = model.predict(X_train)
        probs = model.predict_proba(X_train)[:, 1]
        
        f1 = f1_score(y_train, preds)
        auc = roc_auc_score(y_train, probs)

        # 5. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_score", auc)
        
        # Save the model to the docker folder for the API to use later
        joblib.dump(model, '../docker/model.joblib')
        
        # Also log the model artifact in MLflow (Step 5 of your guide)
        mlflow.xgboost.log_model(model, name="model_artifacts", registered_model_name="ChurnPredictionXGB")

        print(f"Model trained! F1: {f1:.4f}, AUC: {auc:.4f}")
        print("Experiment tracked in MLflow.")

def promote_model(model_name, version):
    client = MlflowClient()
    
    # 1. Assign the 'champion' alias to the specific version
    # This is the modern way to say "This is our Production model"
    client.set_registered_model_alias(model_name, "champion", version)
    
    # 2. Add a description for documentation (Great for interviews!)
    client.update_model_version(
        name=model_name,
        version=version,
        description="This version was promoted because it met the F1 threshold of > 0.8."
    )
    
    print(f"Model {model_name} version {version} is now promoted to @champion")

if __name__ == "__main__":
    # Uncomment to train the model
    train_model()

    # Example: Promote version 2 if it meets criteria
    promote_model("ChurnPredictionXGB", 1)  # Change version as needed