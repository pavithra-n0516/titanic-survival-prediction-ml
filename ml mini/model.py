import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load training data
df = pd.read_csv('data/train.csv')

# Preprocessing
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features to use (simple, known to be predictive)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Fill missing Fare in training (rare)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

X = df[features]
y = df['Survived']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and features list
with open('app/model/titanic_model.pkl', 'wb') as f:
    pickle.dump((model, features), f)

print("Model trained and saved successfully.")
