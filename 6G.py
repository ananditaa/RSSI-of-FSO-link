#import matplotlib as plt
#import numpy as np
import pandas as pd
from sklearn import linear_model
data = pd.read_csv('C:\\Users\\anand\\Documents\\Kolkattadataset.csv')
reg=linear_model.LinearRegression()
reg.fit(data[['Temperature','Relative Humidity','Wind Speed','Wind Direction','Visibility']],data.RSSI)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[34,90,10,95,2]]))             #ans 1.7964dBm 
print(reg.predict([[29.6,85.76,9.2,159.21,2.2]])) #ans 0.91816dBm
print(reg.predict([[43,100,16,174.34,1.2]]))      #ans 3.5928dBm
print(reg.predict([[10,100,100,30,0.5]]))         #ans=-2.994dBm
print(reg.predict([[50,90,20,195,0]]))            #ans=4.99dBm
print(reg.predict([[25,50,5,45,1]]))              #ans=-8.8817842e-16dBm



