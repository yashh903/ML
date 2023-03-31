import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv(r"C:\Users\YASH\Desktop\pandas\Housing.csv")
df.head()
df.tail()

df.columns

df.describe()
df.info()

df.isnull().sum()

sns.boxplot(df['price'])
sns.boxplot(df['area'])
sns.boxplot(df['bedrooms'])
sns.boxplot(df['bathrooms'])
sns.boxplot(df['stories'])
sns.boxplot(df['parking'])

q1=df.price.quantile(0.25)
q3=df.price.quantile(0.75)
iqr=q3-q1
df=df[(df.price >= q1 - 1.5*iqr) & (df.price <= q3 + 1.5*iqr)]

Q1=df.area.quantile(0.25)
Q3=df.area.quantile(0.75)
IQR=Q3-Q1
df=df[(df.area >= Q1 - 1.5*IQR) & (df.area <= Q3 + 1.5*IQR)]

df['mainroad']=df['mainroad'].map({'yes':1,'no':0})
df['guestroom']=df['guestroom'].map({'yes':1,'no':0})
df['basement']=df['basement'].map({'yes':1,'no':0})
df['hotwaterheating']=df['hotwaterheating'].map({'yes':1,'no':0})
df['airconditioning']=df['airconditioning'].map({'yes':1,'no':0})
df['prefarea']=df['prefarea'].map({'yes':1,'no':0})
fs=pd.get_dummies(df['furnishingstatus'])
fs=fs.drop('furnished',axis=1)
df=pd.concat([df,fs],axis=1)
df=df.drop('furnishingstatus',axis=1)

sns.pairplot(df)
sns.jointplot(data=df,x='price',y='area')
sns.jointplot(data=df,x='price',y='area',hue='bedrooms')
sns.jointplot(data=df,x='price',y='area',hue='bathrooms')
sns.jointplot(data=df,x='price',y='area',hue='parking')


num_cols=['price','area','bedrooms','bathrooms','stories','parking']

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(df[num_cols])
df[num_cols]=ss.transform(df[num_cols])

x=df.drop('price',axis=1)
y=df.price

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=55)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
r2_score(y_test, y_pred)
np.sqrt(mean_squared_error(y_test, y_pred))
mean_absolute_error(y_test, y_pred)

plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('actual vs predication')

lr.coef_

lr.intercept_



