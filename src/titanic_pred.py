def load_and_preprocess_test_data():
    import pandas as pd
    import warnings

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
    import mlflow
    model_name = "TitanicModel2"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
    return model

def make_predictions(model, test_df):
    predictions = model.predict(test_df)
    return predictions

def create_submission(test_df, predictions):
    import pandas as pd

    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
    submission.to_csv('../Results/submission.csv', index=False)

def main():
    test_df = load_and_preprocess_test_data()
    model = load_model()
    predictions = make_predictions(model, test_df)
    create_submission(test_df, predictions)

if __name__ == '__main__':
    main()
