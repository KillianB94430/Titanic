

import pandas as pd
import warnings

def load_and_preprocess_data():
    # Charger les données d'entraînement et de test
    train_df = pd.read_csv('../../Data/train.csv')
    test_df = pd.read_csv('../../Data/test.csv')

    # Afficher les colonnes des DataFrames
    print(train_df.columns)
    print(test_df.columns)

    warnings.filterwarnings('ignore')

    # Prétraiter les données
    if 'Embarked' in train_df.columns:
        train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
    else:
        print("La colonne 'Embarked' n'est pas présente dans train_df")

    if 'Embarked' in test_df.columns:
        test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)
    else:
        print("La colonne 'Embarked' n'est pas présente dans test_df")

    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
    test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

    test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

    train_df.drop('Cabin', axis=1, inplace=True)
    test_df.drop('Cabin', axis=1, inplace=True)

    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

    train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'])
    test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'])

    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

    for feature in features:
        if feature not in train_df.columns:
            train_df[feature] = 0
        if feature not in test_df.columns:
            test_df[feature] = 0

    X = train_df[features]
    y = train_df['Survived']

    # Sauvegarder les données prétraitées pour utilisation ultérieure
    X.to_csv('../../Data/X_train.csv', index=False)
    y.to_csv('../../Data/y_train.csv', index=False)
    test_df.to_csv('../../Data/test_df.csv', index=False)

