# load_model_and_predict.py

import pandas as pd
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def load_model_and_predict():
    # Charger les données de test prétraitées
    test_df = pd.read_csv('../../Data/test_df.csv')

    # Initialiser MlflowClient et rechercher le modèle
    client = MlflowClient()
    model_name = "TitanicModel2"
    filter_string = f"name='{model_name}'"
    registered_versions = client.search_registered_models(filter_string=filter_string, order_by=(["last_updated_timestamp DESC"]))
    print(registered_versions)
    last_model_uri = registered_versions[0].latest_versions[-1].source    
    print(f"last_model_uri : {last_model_uri}")

    # Charger le modèle depuis MLflow
    loaded_model = mlflow.sklearn.load_model(last_model_uri)

    # Faire des prédictions sur le jeu de test
    X_test = test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    predictions = loaded_model.predict(X_test)

    # Générer le fichier de soumission
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
    submission.to_csv('submission_pred.csv', index=False)

    print("Les prédictions ont été enregistrées dans submission_pred.csv")
