import pandas as pd
df=pd.read_csv(r"C:\Users\YASH\Desktop\pandas\loan_data_set.csv")
test=pd.read_csv(r"C:\Users\YASH\Desktop\pandas\test_Y3wMUE5_7gLdaTN.csv")
df.isnull().sum()
test.isnull().sum()
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].median())


test['Gender']=test['Gender'].fillna(test['Gender'].mode()[0])
test['Dependents']=test['Dependents'].fillna(test['Dependents'].mode()[0])
test['Self_Employed']=test['Self_Employed'].fillna(test['Self_Employed'].mode()[0])
test['LoanAmount']=test['LoanAmount'].fillna(test['LoanAmount'].median())
test['Loan_Amount_Term']=test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].median())
test['Credit_History']=test['Credit_History'].fillna(test['Credit_History'].median())
df.columns
df=df.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

x=df.drop('Loan_Status',axis=1)
y=df.Loan_Status

df=pd.get_dummies(df)
x=pd.get_dummies(x)
test=pd.get_dummies(test)

from sklearn.model_selection import train_test_split
x_train,x_v,y_train,y_v=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
model.fit(x_train,y_train)

predication=model.predict(x_v)
accuracy_score(y_v,predication)
mew_pred=model.predict(test)





















