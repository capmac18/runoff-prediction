import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
from statistics import mean
from sklearn.metrics import r2_score

import camelot

style.use("ggplot")

def squared_error(y_line,y_point):
	return sum((y_line-y_point)**2)

def coeff_of_determination(y_point,y_line):
	return 1-squared_error(y_line,y_point)/squared_error(y_line,mean(y))

file = "Ozarkhed_Rainfall_Runoff.pdf"

tables = camelot.read_pdf(file)

df = tables[0].df
# print(df)

df.drop(df.index[11],inplace=True)
#print(df)
df=df[1:]
#data_list = df.astype(float).values.tolist()
#print(df)

year= df[0].astype(int).values.tolist()          #since by default the dataframe stores as string 
X = np.array(df[1]).astype(float)
y = np.array(df[2]).astype(float)
#print(year)
#print(X)
#print(y)

train_size = int(len(X)*0.8)

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state=0)
X_train = X[:train_size]
X_train = X_train.reshape(len(X_train),-1)
y_train = y[:train_size]
#y_train = y_train.reshape(1,-1)
X_test = X[train_size:]
X_test = X_test.reshape(len(X_test),-1)
y_test = y[train_size:]
#y_test = y_test.reshape(1,-1)

#print(X_test,y_test)
classifier = LinearRegression()
classifier.fit(X_train,y_train)
y_line = classifier.predict(X_train)

accuracy = classifier.score(X_test,y_test)
r_square = coeff_of_determination(y_train,y_line)   
r_squared_auto = r2_score(y_test, classifier.predict(X_test))												

#print("Accuracy :",accuracy)
print("R_squared ML model :", r_squared_auto)
#print("R_squared_inbuilt :", r_squared_auto)

# # line from pdf
# y_line_pdf = 0.89651 * X_train.flatten() - 541.18656
# r_squared_pdf = r2_score(y_train, y_line_pdf)	
# print("R_squared_inbuilt_pdf :", r_squared_pdf)

#line from Inglis formula
#X_inglis = np.array([i for i in X if(i>=2000)])
#y_inglis = np.array([y[i]  for i in range(len(X)) if(X[i]>=2000)])
#y_line_inglis = 0.85 * X_inglis - 304.8
X_train_extended = np.array([[1000.0]] + list(X_train) + [[3200.0]], dtype=np.float32)
#print(X_train_extended.shape)

y_line_inglis = 0.85 * X_train_extended - 304.8
y_inglis_pred = 0.85 * X_test - 304.8
r_squared_inglis = r2_score(y_test, y_inglis_pred)
print("R_squared_inbuilt_inglis :", r_squared_inglis)

plt.scatter(X_train, y_train, label='train data')
plt.scatter(X_test,y_test,color='b', label='test data')



plt.plot(X_train_extended, classifier.predict(X_train_extended), color='tab:purple', label=f'ML model | R2 = {r_squared_auto:.2f}')
#plt.plot(X_train,y_line_pdf,color='g')
plt.plot(X_train_extended, y_line_inglis, color='tab:orange', label=f'Inglis | R2 = {r_squared_inglis:.2f}')
plt.xlabel("Rainfall in mm")
plt.ylabel("Runoff in mm")
plt.legend()
plt.show()