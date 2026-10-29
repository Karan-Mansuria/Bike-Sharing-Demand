import pandas as pd
import numpy as np

print("Loading datasets...")
train= pd.read_csv("train.csv",parse_dates=["datetime"])
test= pd.read_csv("test.csv",parse_dates=["datetime"])

def preprocessing(data):
    datacopy=data.copy()
    datacopy['hour']=data['datetime'].dt.hour
    datacopy['day']=data['datetime'].dt.dayofweek
    datacopy['month']=data['datetime'].dt.month
    datacopy['year']=data['datetime'].dt.year

    features=['hour','day','month','year','season','holiday','workingday','weather','temp','atemp','humidity','windspeed']
    X=datacopy[features].copy()
    return X

X=preprocessing(train)
y=train['count']

X_test=X.sample(frac=0.2,random_state=67)
y_test=y.loc[X_test.index]

X_train=X.drop(X_test.index)
y_train=y.loc[X_train.index]

X_train_np=X_train.values
y_train_np=y_train.values

X_test_np=X_test.values
y_test_np=y_test.values

X_test_np_final=np.hstack([np.ones((X_test_np.shape[0],1)),X_test_np])

X_train_np_final = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])  

w=np.linalg.inv(X_train_np_final.T@X_train_np_final)@(X_train_np_final.T@y_train_np)

y_predicted=X_test_np_final@w

mse=np.mean((y_test_np-y_predicted)**2)
print("MSE: " ,mse)

ssres=np.sum((y_test_np-y_predicted)**2)
sstot=np.sum((y_test_np-np.mean(y_test_np))**2)
r2=1-ssres/sstot
print("R^2",r2)


