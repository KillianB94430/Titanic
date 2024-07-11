def load_and_preprocess_data():
    import pandas as pd
    import warnings

    train_df = pd.read_csv('../Data/train.csv')
    warnings.filterwarnings('ignore')

    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
    train_df.drop('Cabin', axis=1, inplace=True)
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    train_df['IsAlone'] = 1
    train_df.loc[train_df['FamilySize'] > 1, 'IsAlone'] = 0
    train_df = train_df.drop(['Name', 'Ticket', 'SibSp', 'Parch'], axis=1)
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    train_df = pd.get_dummies(train_df, columns=['Embarked'])

    X = train_df.drop('Survived', axis=1)
    y = train_df['Survived']
    return X, y

def train_model(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_val, y_val

def evaluate_model(model, X_val, y_val):
    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {accuracy}')
    return accuracy

def save_model(model, accuracy):
    import mlflow
    from mlflow import MlflowClient

    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('random_state', 42)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, 'model')

    if accuracy > 0.84:
        model_uri = f'runs:/{mlflow.active_run().info.run_id}/model'
        print(f'Model URI: {model_uri}')
        model_name = "TitanicModel2"

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

def main():
    X, y = load_and_preprocess_data()
    model, X_val, y_val = train_model(X, y)
    accuracy = evaluate_model(model, X_val, y_val)
    save_model(model, accuracy)

if __name__ == '__main__':
    main()
