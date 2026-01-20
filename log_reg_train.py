import pandas as pd
import numpy as np
import pickle

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score




# laod dataset

df = pd.read_csv("train.csv")


# remove unnecessary columns
feature_to_drop = ['three_g', 'fc']

if 'id' in df.columns:
  feature_to_drop.append('id')

for col in feature_to_drop:
    df.drop(col, axis=1, inplace=True)


# replace 0 with median for Screen width 
df['sc_w'] = df['sc_w'].replace(0, df.loc[df['sc_w'] != 0, 'sc_w'].median())
(df['sc_w'] == 0).sum()


# split feature and target
X = df.drop('price_range', axis=1)
y = df['price_range']


# column split
numeric_feature = X.select_dtypes(include=['int64', 'float64']).columns
categorical_feature = X.select_dtypes(include=['object']).columns


# num 
num_tranformer = Pipeline(
    steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

# cat
cat_transformer = Pipeline(
    steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_tranformer, numeric_feature),
        ('cat', cat_transformer, categorical_feature)
    ]
)

# Logistic Regression
lg_reg = LogisticRegression(
    C=100,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', lg_reg)
    ]
)

#  train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# evaluation
y_pred = pipeline.predict(X_test)

# matrix
final_accuracy = accuracy_score(y_test, y_pred)
final_f1 = f1_score(y_test, y_pred, average='weighted')
final_report = classification_report(y_test, y_pred)

print("Final Accuracy:", final_accuracy)
print("Final F1 Score:", final_f1)
print("Final Classification Report:\n", final_report)


with open('lg_reg_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print('Logistic Regression model saved as lg_reg_pipeline.pkl')