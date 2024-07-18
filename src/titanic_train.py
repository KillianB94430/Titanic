import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow import MlflowClient

def load_data():
    """
    Load the dataset from a CSV file.

    Args:
        filepath (str): The path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """
    return pd.read_csv("../Data/train.csv")

def preprocess_data(train_df):
    """
    Preprocess the dataset by handling missing values, creating new features,
    and dropping unnecessary columns.

    Args:
        train_df (pd.DataFrame): The raw dataset.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    warnings.filterwarnings("ignore")  # Ignore warnings
    # Fill missing values
    train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
    train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)

    # Drop the 'Cabin' column due to high percentage of missing values
    train_df.drop("Cabin", axis=1, inplace=True)

    # Create new feature 'FamilySize'
    train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1

    # Create new feature 'IsAlone'
    train_df["IsAlone"] = 1
    train_df.loc[train_df["FamilySize"] > 1, "IsAlone"] = 0

    # Drop unnecessary columns
    train_df = train_df.drop(["Name", "Ticket", "SibSp", "Parch"], axis=1)

    # Convert 'Sex' column to numerical values
    train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1}).astype(int)

    # One-hot encode 'Embarked' column
    train_df = pd.get_dummies(train_df, columns=["Embarked"])

    return train_df

def split_data(train_df):
    """
    Split the dataset into features (X) and target (y).

    Args:
        train_df (pd.DataFrame): The preprocessed dataset.

    Returns:
        pd.DataFrame, pd.Series: The features (X) and target (y).
    """
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]
    return X, y

def train_model(X, y):
    """
    Train a random forest model on the training data.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.

    Returns:
        RandomForestClassifier: The trained random forest model.
        pd.DataFrame: The validation feature matrix.
        pd.Series: The validation target vector.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_val, y_val

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the trained model on the validation data.

    Args:
        model (RandomForestClassifier): The trained random forest model.
        X_val (pd.DataFrame): The validation feature matrix.
        y_val (pd.Series): The validation target vector.

    Returns:
        float: The accuracy score of the model on the validation data.
    """
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {accuracy}')
    return accuracy
    
def set_mlflow_experiment():
    """
    Set up and run the MLflow experiment.
    """
    
    with mlflow.start_run():
        set_mlflow_experiment("Titanic")

def save_model(model, accuracy,n_estimators=100,random_state=42):
    """
    log parameters and metrics to MLflow.

    Args:
        n_estimators (int): Number of estimators for the model.
        random_state (int): Random state for reproducibility.

    Returns:
    """
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('random_state', random_state)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, 'model')

def save_model_with_check(model,accuracy,treshold=0.84):
    """
    Save the trained model to disk and register it with MLflow if the accuracy exceeds the threshold.

    Args:
        model : The trained  model.
        accuracy (float): The accuracy score of the model.
        threshold (float): The accuracy threshold for model registration.
    """
    
    if accuracy >treshold:
        model_uri = f'runs:/{mlflow.active_run().info.run_id}/model'
        print(f'Model URI: {model_uri}')
        model_name = 'TitanicModel2'

        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException as e:
            print(f"Registered model '{model_name}' already exists or another error occurred: {e}")

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id
        )
        print(f'Model version {model_version.version} with run_id {mlflow.active_run().info.run_id} is registered to production.')
    else:
        print('Model accuracy is below the threshold. Model not registered.')








def run_pipeline_train():
    """
    Run the full training pipeline: load data, preprocess, train, evaluate, and save the model.
    """
    # Load and preprocess the data
    train_df= load_data()
    train_df=preprocess_data(train_df) 

     # Split the data into features and target
    X, y = split_data(train_df)

    # Train the model
    model, X_val, y_val = train_model(X, y)

    # Evaluate the model
    accuracy = evaluate_model(model, X_val, y_val)

    # Save the model with MLflow logging
    save_model(model, accuracy)

    # Check accuracy and save model if it meets the threshold
    save_model_with_check(model, accuracy)

if __name__ == '__main__':
    run_pipeline_train()
