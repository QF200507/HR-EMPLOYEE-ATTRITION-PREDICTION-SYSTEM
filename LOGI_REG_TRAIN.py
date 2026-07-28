import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib

df=pd.read_csv("hr_emp2.csv")

x=df.drop("Attrition",axis=1)
y=df["Attrition"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)

scale=StandardScaler()
x_train=scale.fit_transform(x_train)

x_test=scale.transform(x_test)

print(type(x_train))

model=LogisticRegression(random_state=42)
model.fit(x_train,y_train)

print(model.coef_.shape)

y_pred=model.predict(x_test)

prob=model.predict_proba(x_test)
print(prob[:10])

custom_threshold = 0.21
y_pred_custom = (prob[:, 1] >= custom_threshold).astype(int)
print(y_pred_custom)

accuracy = accuracy_score(y_test, y_pred_custom)
print(accuracy)
print(classification_report(y_test, y_pred_custom))

joblib.dump({"model":model,"scaler": scale,"feature_columns":x.columns.tolist(),"default_threshold":0.21},"model.pk1")