import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('heart.csv')
#print(df.head())
#print(df.info())
cat_cols = [cols for cols in df.columns if df[cols].dtype == 'object']
num_cols = [cols for cols in df.columns if df[cols].dtype != 'object']
'''print(cat_cols)
print(num_cols)'''
#cat_cols1=pd.DataFrame(cat_cols)
#cat_cols=cols for cols in cat_cols.columns pd.to_numeric(cat_cols[cols])]
'''for cols in cat_cols:
   df[cols] = df[cols].astype('Int64',errors='coerce')'''

#preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
num_data = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

# Convert each categorical column to integer using LabelEncoder
cat_data = pd.DataFrame()
le = LabelEncoder()
for col in cat_cols:
    cat_data[col] = le.fit_transform(df[col])

df1=pd.concat([num_data,cat_data],axis=1)
df2=df1.copy()
df2[['target']].astype(int)
#print(df1['target'])
'''print(df1.head())
print(df1.info())'''
df2[['sex']].astype(int)
from sklearn.model_selection import train_test_split
X=df2[['age', 'sex', 'cp', 'oldpeak', 'trestbps', 'chol', 'fbs',
       'restecg', 'thalachh', 'exang', 'slope', 'ca', 
'thal']]
Y=df2[['target']].astype(int)

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=100)

from sklearn.linear_model import LogisticRegression

regressor=LogisticRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,Y_pred))

plt.figure(figsize=(8,6))
cm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm,annot=True,cmap='Blues',fmt="d",xticklabels=["Fail","Pass"],yticklabels=["Fail","Pass"])
plt.show()
