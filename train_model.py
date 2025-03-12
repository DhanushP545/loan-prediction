import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("data/loan_train.csv")
numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
for col in numeric_cols:
    df[col]= df[col].fillna(df[col].median())

encoder = LabelEncoder()
cols = ["Loan_ID","Gender", "Married", "Dependents","Education", "Self_Employed", "Property_Area", "Loan_Status"]
for col in cols:
    df[col] = encoder.fit_transform(df[col])
    #print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))) # to check encoding

df = df.fillna(0)

# no normalization required , we are using random forest

#training

X = df.drop(columns=["Loan_Status", "Loan_ID","Gender", "Married", "Dependents","Education", "Self_Employed"])
Y = df['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=100,  # Reduce number of trees
    max_depth=10,      # Limit tree depth to avoid overfitting
    min_samples_split=5,  # Require more samples to split a node
    max_features='sqrt',  # Use only a subset of features per split
    random_state=42
)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print("accuracy: ", accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

joblib.dump(model, "models/loan_approval_01.pkl")
print("saved")