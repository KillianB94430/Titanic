

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime


# Charger les données prétraitées
X = pd.read_csv('../../Data/X_train.csv')
y = pd.read_csv('../../Data/y_train.csv')


# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir l'expérience MLflow
mlflow.set_experiment('Titanic')

with mlflow.start_run() as run:
    run_name = datetime.now().strftime("Run_%Y-%m-%d_%H-%M-%S")
    client = MlflowClient()
    client.set_tag(run.info.run_id, "mlflow.runName", run_name)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {accuracy}')
    
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('random_state', 42)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, 'model')
    
    if accuracy > 0.84:
        model_uri = f'runs:/{mlflow.active_run().info.run_id}/model'
        print(f'Model URI: {model_uri}')
        
        model_name = "TitanicModel2"
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException as e:
            print(f"Registered model '{model_name}' already exists or another error occurred: {e}")
        
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run.info.run_id
        )

        print(f'Model version {model_version.version} with run_id {run.info.run_id} is registered to production.')
    else:
        print('Model accuracy is below the threshold. Model not registered.')
