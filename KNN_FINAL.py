import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


df=pd.read_csv("C:\\Users\\anand\\Documents\\Kolkattadata1.csv")
X=df[['Temperature','Relative Humidity','Wind Speed','Wind Direction','Visibility']]
Y=df['RSSI']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.6,random_state=0)

clf=KNeighborsRegressor(11)
clf.fit(X_train,Y_train)

Y_predict=clf.predict(X_test)
print(mean_squared_error(Y_test,Y_predict))

error=[]
for k in range(1,15):
    knn=KNeighborsRegressor(n_neighbors=k)
    Y_predict=cross_val_predict(knn,X_test,Y_test,cv=5)
    error.append(mean_squared_error(Y_test,Y_predict))

plt.plot(range(1,15),error)
plt.show(block=True)
