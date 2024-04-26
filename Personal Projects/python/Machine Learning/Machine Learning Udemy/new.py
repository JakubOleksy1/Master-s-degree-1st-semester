import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

os.chdir("C:/Users/jakub/Visual Studio Code/Personal Projects/python/Machine Learning/Machine Learning Udemy")

df = pd.read_csv('Obesity Classification.csv')

df = df.drop('ID', axis=1)
print(df.isnull().sum())

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Label'] = le.fit_transform(df['Label'])

print(df['Gender'].unique())
print(df['Label'].unique())

correlations = df.corr().abs()

X = df.drop(['Label'], axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = DecisionTreeClassifier()
print(X_train.shape)
print(y_train.shape)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))