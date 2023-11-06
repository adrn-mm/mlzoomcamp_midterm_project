import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Importing data
# Note: Replace 'your_data.csv' with the path to your dataset file
raw_data_path = os.path.join(os.getcwd(), 'data','raw_data.csv')
df = pd.read_csv(raw_data_path)

# Data transformation and feature engineering here
# One-hot encode the 'type' column
df = pd.get_dummies(df, columns=['Type'], prefix='', prefix_sep='')

# Convert the 'product ID' to numeric by extracting the numbers and categorizing the quality
df['product_quality'] = df['Product ID'].str.extract('([LMH])')[0]
df['product_serial'] = df['Product ID'].str.extract('(\d+)').astype(int)
df = df.drop('Product ID', axis=1)

# Splitting the data
X = df.drop(['UDI', 'Machine failure'], axis=1) # drop the target and identifier columns
y = df['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = ['product_quality']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Logistic Regression model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000, random_state=42))])

# Training the model
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Saving the model
saved_model_path = os.path.join(os.getcwd(), 'model', 'logistic_regression_model.joblib')
joblib.dump(model, saved_model_path)
print("Model saved as {}".format(saved_model_path))