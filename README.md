# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-1:
Import the required libraries.

Step-2:
Load the dataset.

Step-3:
Define X and Y array.

Step-4:
Define a function for costFunction,cost and gradient.

Step-5:
Define a function to plot the decision boundary.

step 6:
Define a function to predict the Regression value.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:S.Suruthi
RegisterNumber:212224220114  
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
print(x.head())

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![Screenshot 2025-04-25 003902](https://github.com/user-attachments/assets/979a982b-d149-4272-b140-34ea3a0f5ad8)
![Screenshot 2025-04-25 004011](https://github.com/user-attachments/assets/6aaac30f-0d4f-467f-ab91-a2fd5dc36c3c)
![Screenshot 2025-04-25 004232](https://github.com/user-attachments/assets/f47eb0e1-f757-4329-be73-1900f98443df)

## Accuracy:

![image](https://github.com/user-attachments/assets/09b9f444-3b6b-4224-881e-55235c9a2c9e)

## Prediction:

![image](https://github.com/user-attachments/assets/549ba1d3-86ab-484f-8f5e-55584928c277)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
