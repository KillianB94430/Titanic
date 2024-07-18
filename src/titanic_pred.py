import warnings
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

def load_data(filepath="../Data/test.csv"):
    """
    Load the dataset from a CSV file.

    Args:
        filepath (str): The path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """
    return pd.read_csv(filepath)

def preprocess_data(test_df):
    """
    Preprocess the dataset by handling missing values, creating new features,
    and dropping unnecessary columns.

    Args:
        test_df (pd.DataFrame): The raw dataset.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    warnings.filterwarnings("ignore")  # Ignore warnings
    # Fill missing values
    test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
    test_df["Embarked"].fillna(test_df["Embarked"].mode()[0], inplace=True)
    test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

    # Drop the 'Cabin' column due to high percentage of missing values
    test_df.drop("Cabin", axis=1, inplace=True)

    # Create new feature 'FamilySize'
    test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

    # Create new feature 'IsAlone'
    test_df["IsAlone"] = 1
    test_df.loc[test_df["FamilySize"] > 1, "IsAlone"] = 0

    # Drop unnecessary columns
    test_df = test_df.drop(["Name", "Ticket", "SibSp", "Parch"], axis=1)

    # Convert 'Sex' column to numerical values
    test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1}).astype(int)

    # One-hot encode 'Embarked' column
    test_df = pd.get_dummies(test_df, columns=["Embarked"])

    return test_df  # Return the preprocessed DataFrame

def load_model(model_name="TitanicModel2"):
    """
    Load the latest version of the specified model from MLflow.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        sklearn.base.BaseEstimator: The loaded model.
    """
    client = MlflowClient()
    filter_string = f"name='{model_name}'"
    results = client.search_registered_models(filter_string=filter_string)

    if not results:
        raise ValueError(f"No registered models found with name '{model_name}'")
    
    # Print the results for debugging purposes
    print("Search registered models results:")

    for res in results:
        for mv in res.latest_versions:
            print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

    # Get the latest version of the model
    registered_versions = client.search_registered_models(
        filter_string=filter_string, order_by=["last_updated_timestamp DESC"]
    )

    last_model_uri = registered_versions[0].latest_versions[-1].source

    # Load the appropriate model identified above
    loaded_model = mlflow.sklearn.load_model(last_model_uri)
    return loaded_model

def make_predictions(model, test_df):
    """
    Use the loaded model to make predictions on the test dataset.

    Args:
        model (sklearn.base.BaseEstimator): The loaded model.
        test_df (pd.DataFrame): The preprocessed test dataset.

    Returns:
        np.ndarray: The predictions.
    """
    predictions = model.predict(test_df)
    return predictions

def create_submission(test_df, predictions):
    """
    Create a CSV file for the submission.

    Args:
        test_df (pd.DataFrame): The preprocessed test dataset.
        predictions (np.ndarray): The predictions.

    Returns:
        None
    """ 
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
    submission.to_csv('../Results/submission_pred.csv', index=False)

def run_pipeline_pred():
    """
    Run the full prediction pipeline: load data, preprocess, load model,
    make predictions, and create submission.
    """
    # Load and preprocess the test data
    test_df = load_data()
    test_df = preprocess_data(test_df)
    
    
    # Load the trained model
    model = load_model()
    
    # Make predictions on the test data
    predictions = make_predictions(model, test_df)
    
    # Create a submission file
    create_submission(test_df, predictions)

if __name__ == '__main__':
    run_pipeline_pred()
