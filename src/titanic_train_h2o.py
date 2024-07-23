import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
import h2o
from h2o.estimators import H2ORandomForestEstimator

def load_data():
    """
    Load the dataset from a CSV file.

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
    Train a Random Forest model on the training data using H2O.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.

    Returns:
        H2ORandomForestEstimator: The trained H2ORandomForestEstimator model.
        pd.DataFrame: The validation feature matrix.
        pd.Series: The validation target vector.
    """
    # Start H2O cluster
    h2o.init()

    # Convert pandas DataFrame to H2O Frame
    train_h2o = h2o.H2OFrame(pd.concat([X, y], axis=1))
    
    # Define the target and features
    target = 'Survived'
    features = [col for col in train_h2o.columns if col != target]

    # Split the dataset into training and validation sets
    train, valid = train_h2o.split_frame(ratios=[0.8], seed=42)
    
    # Initialize H2O Random Forest
    rf_model = H2ORandomForestEstimator(ntrees=100, seed=42)
    
    # Train the model
    rf_model.train(x=features, y=target, training_frame=train, validation_frame=valid)
    
    return rf_model, valid[features], valid[target]

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the trained model on the validation data.

    Args:
        model (H2ORandomForestEstimator): The trained H2ORandomForestEstimator model.
        X_val (H2OFrame): The validation feature matrix.
        y_val (H2OFrame): The validation target vector.

    Returns:
        float: The accuracy score of the model on the validation data.
    """
    # Predict using the model
    y_pred = model.predict(X_val)
    y_pred = y_pred.as_data_frame()['predict'].values
    y_val = y_val.as_data_frame().values.ravel()

    # Convert predictions to binary values
    y_pred = (y_pred > 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {accuracy}')
    
    return accuracy

def save_model(model, accuracy, threshold=0.84):
    """
    Save the trained model and its performance metrics using H2O.

    Args:
        model (H2ORandomForestEstimator): The trained H2ORandomForestEstimator model.
        accuracy (float): The accuracy score of the model.
        threshold (float): The accuracy threshold for saving the model.
    """
    if accuracy > threshold:
        # Save the model
        model_path = f'h2o_model_{accuracy:.2f}.zip'
        h2o.save_model(model=model, path=model_path, force=True)
        print(f'Model saved to {model_path}')
        print(f"Model ID: {model.model_id}")
    else:
        print('Model accuracy is below the threshold. Model not saved.')

def run_pipeline_train():
    """
    Run the full training pipeline: load data, preprocess, train, evaluate, and save the model.
    """
    # Load and preprocess the data
    train_df = load_data()
    train_df = preprocess_data(train_df)

    # Split the data into features and target
    X, y = split_data(train_df)

    # Train the model
    model, X_val, y_val = train_model(X, y)

    # Evaluate the model
    accuracy = evaluate_model(model, X_val, y_val)

    # Save the model with H2O
    save_model(model, accuracy)

if __name__ == '__main__':
    run_pipeline_train()



