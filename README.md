# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 2.Set variables for assigning dataset values. 3.Import linear regression from sklearn. 4.Assign the points for representing in the graph. 5.Predict the regression for marks by using the representation of the graph. 6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Stephen raj.Y
RegisterNumber: 212223230217
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

## Dataset:

![image](https://github.com/user-attachments/assets/e5bcb547-05a9-4ad3-9d27-7a27e250eb53)

## Head values:

![image](https://github.com/user-attachments/assets/67a8d0d0-f87d-4817-b7e3-5c2d4dfd099a)

## Tail values:

![image](https://github.com/user-attachments/assets/65869270-28b9-45d4-8a66-230d8aa712f4)

## X and Y values:

![image](https://github.com/user-attachments/assets/fa3ea1ce-2776-4014-8966-f8a32639f0d1)

## Predication values of X and Y:

![image](https://github.com/user-attachments/assets/51f63053-dc56-42f6-9548-e4dc9862b2ed)

## MSE,MAE and RMSE:

![image](https://github.com/user-attachments/assets/e84db591-cf6f-4e9f-99ab-bc96640386ec)

## Training Set:

![image](https://github.com/user-attachments/assets/1a7ee788-6137-4ad7-a41a-9d875e623aed)

## Testing Set:

![image](https://github.com/user-attachments/assets/86609676-8b3a-437d-bbb6-de22157c2290)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
