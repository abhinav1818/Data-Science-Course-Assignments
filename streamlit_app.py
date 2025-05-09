import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

df= pd.read_csv("Titanic_train.csv")


#EDA
df.describe()

#Histogram

import matplotlib.pyplot as plt
import seaborn as sns

# # Visualize distribution of each column using histograms
# for col in df.columns:
#     plt.figure(figsize=(4, 2))
#     sns.histplot(df[col])
#     plt.title(f"Histogram of {col}")
#     plt.show()

 

#Data Preprocessing

df.isnull().sum()
df["Age"].fillna(df["Age"].median(),inplace = True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace = True)
#most of cabin variable contains missing values so droping the variable
df.drop(columns=['Cabin','Name','Ticket','PassengerId'],inplace = True)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)



#data transformation

#applying log transformation on highly skewed data variables

#for right-skewed data
columns_to_log = ['Fare','Parch','SibSp']  # Replace with your actual columns
for col in columns_to_log:
    if col == 'Fare':
        df[col] = np.log1p(df[col])
    else:
        df[col] = np.log(df[col] + 1)



# dATA TRANSFORMTION

#label-encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder() 

#1- Female, 2 - Petrol
df["Sex"] = LE.fit_transform(df["Sex"])

#standardizing the data

cols_to_scale = ['Fare', 'Age']

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
for col in cols_to_scale:
    df[col] = SS.fit_transform(df[[col]])

#Data partition

from sklearn.model_selection import train_test_split

# Split data
X = df.drop('Survived', axis=1)
Y = df['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)


#model fitting
from sklearn.linear_model import LogisticRegression

# Train logistic regression
model = LogisticRegression(max_iter=210)
model.fit(X_train, y_train)

#model prediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:,1]

# #Interpretation of Co-effcients
# coef = pd.Series(model.coef_[0], index=X_train.columns).sort_values()
# print(coef)

#Data Preprocessing for test data

def preprocess(df_test):
    df_test["Age"].fillna(df_test["Age"].median(), inplace=True)
    df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)
    if 'Cabin' in df_test.columns:
        df_test.drop(columns=['Cabin'], inplace=True)
    if 'Name' in df_test.columns:
        df_test.drop(columns=['Name'], inplace=True)
    if 'Ticket' in df_test.columns:
        df_test.drop(columns=['Ticket'], inplace=True)
    if 'PassengerId' in df_test.columns:
        df_test.drop(columns=['PassengerId'], inplace=True)
    
    # Encode 'Embarked' only if it exists
    if 'Embarked' in df_test.columns:
        df_test["Embarked"].fillna(df_test["Embarked"].mode()[0], inplace=True)
        df_test = pd.get_dummies(df_test, columns=['Embarked'], drop_first=True)
    
    return df_test

def transform(df):
    #1- Female, 2 - Petrol
    df["Sex"] = LE.fit_transform(df["Sex"])
    
    #standardizing the data
    cols_to_scale = ['Fare', 'Age']
    for col in cols_to_scale:
        df[col] = SS.fit_transform(df[[col]])
    return df




import streamlit as st
# Define user input UI
st.title("Titanic Survival Prediction")

sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
fare = st.slider("Fare", 0.0, 600.0, 32.2)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# When user clicks "Predict"
if st.button("Predict Survival"):

    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [LE.transform([sex])[0]],
        "Age": SS.transform([[age]])[0],
        "SibSp": [np.log(sibsp + 1)],
        "Parch": [np.log(parch + 1)],
        "Fare": np.log1p([fare]),
        "Embarked_Q": [1 if embarked == "Q" else 0],
        "Embarked_S": [1 if embarked == "S" else 0]
    })

    # Ensure all expected features are present (matching training data)
    expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Add any missing columns with 0

    input_data = input_data[expected_columns]  # Reorder columns

    # Predict
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        st.write("Survived" if prediction == 1 else "Did Not Survive")
        st.write(f"Survival Probability: {probability:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
