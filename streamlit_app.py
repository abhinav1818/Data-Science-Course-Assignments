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
st.subheader("Upload CSV File for Prediction")
uploaded_file = st.file_uploader("Titanic_test.py",type=["csv"])

if uploaded_file:
    try:
        
        user_df = pd.read_csv(uploaded_file,on_bad_lines='skip')
        st.write("File uploaded successfully!")
        # Check if user_df is None or not
        if user_df is None:
            st.error("The uploaded DataFrame is None.")
            st.stop()

        st.write("First few rows of uploaded data:", user_df.head())
        original = user_df.copy()
        
        # Check for target column
        has_target = 'Survived' in user_df.columns
    
        # Preprocess user data
        user_df = preprocess(user_df)
        user_df = transform(user_df)
        
    
        # Align columns after preprocessing
        missing_cols = set(X.columns) - set(user_df.columns)
        for col in missing_cols:
            user_df[col] = 0  # add missing columns with default value
        user_df = user_df[X.columns]  # reorder to match training
    
        if has_target:
            X_user = user_df.drop("Survived", axis=1)
            y_user = original['Survived']
            y_pred = model.predict(X_user)
            y_prob = model.predict_proba(X_user)[:, 1]
    
            # Evaluation metrics
            st.write("**Uploaded Data Evaluation:**")
            st.write("Accuracy:", accuracy_score(y_user, y_pred))
            st.write("Precision:", precision_score(y_user, y_pred))
            st.write("Recall:", recall_score(y_user, y_pred))
            st.write("F1 Score:", f1_score(y_user, y_pred))
            st.write("ROC AUC Score:", roc_auc_score(y_user, y_prob))
    
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_user, y_prob)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label='ROC Curve')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve (Uploaded File)")
            ax.legend()
            st.pyplot(fig)
        else:
            # Predict for unlabeled data
            y_pred = model.predict(user_df)
            y_prob = model.predict_proba(user_df)[:, 1]
            result_df = original.copy()
            result_df['Prediction'] = y_pred
            result_df['Survival Probability'] = y_prob
            st.write("### Prediction Results")
            st.dataframe(result_df)
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", data=csv, file_name="predictions.csv", mime='text/csv')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.warning("Please upload a CSV file to proceed.")
