import pandas as pd
import warnings
import mlflow
from mlflow.tracking import MlflowClient

def load_and_preprocess_test_data():
    test_df = pd.read_csv('../Data/test.csv')
    warnings.filterwarnings('ignore')

    test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
    test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)
    test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
    test_df.drop('Cabin', axis=1, inplace=True)
    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
    test_df['IsAlone'] = 1
    test_df.loc[test_df['FamilySize'] > 1, 'IsAlone'] = 0
    test_df = test_df.drop(['Name', 'Ticket', 'SibSp', 'Parch'], axis=1)
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    test_df = pd.get_dummies(test_df, columns=['Embarked'])
    return test_df

def load_model():
    client = MlflowClient()
    model_name = "TitanicModel2"
    filter_string = f"name='{model_name}'"
    results = client.search_registered_models(filter_string=filter_string)
    
    if not results:
        raise ValueError(f"No registered models found with name '{model_name}'")
    
    print(results)
    print("-" * 80)
    for res in results:
        for mv in res.latest_versions:
            print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")
    
    registered_versions = client.search_registered_models(filter_string=filter_string,
        order_by=["last_updated_timestamp DESC"])
    
    print(registered_versions)
    last_model_uri = registered_versions[0].latest_versions[-1].source    
    print(f"last_model_uri : {last_model_uri}")    
    
    # Load the appropriate model identified above    
    loaded_model = mlflow.sklearn.load_model(last_model_uri)
    return loaded_model

def make_predictions(model, test_df):
    predictions = model.predict(test_df)
    return predictions

def create_submission(test_df, predictions):
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
    submission.to_csv('../Results/submission_pred.csv', index=False)

def main():
    test_df = load_and_preprocess_test_data()
    model = load_model()
    predictions = make_predictions(model, test_df)
    create_submission(test_df, predictions)

if __name__ == '__main__':
    main()

