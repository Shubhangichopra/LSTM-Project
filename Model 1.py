#!/usr/bin/env python
# coding: utf-8

# In[2]:


##Importing libraries
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pandas_datareader as pdr


# In[3]:


df= pdr.DataReader('INFY.BO', data_source='yahoo', start='2012-01-01', end='2021-01-01')
df.head()


# In[4]:


##preprocessing data by only using one variable of the data 
## Using only the Close Price Data
df1 = df['Close']


# In[5]:


df1.head()


# In[6]:


df1.shape


# In[7]:


##ploting the data
plt.plot(df1)


# In[8]:


##Rescaling Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(np.array(df1).reshape(-1,1))

scaled_data


# In[9]:


scaled_data.shape


# In[10]:


##spliting the data at 85%
training_size = int(len(scaled_data)*0.85)
test_size = len(scaled_data)-training_size

train_data,test_data = scaled_data[0:training_size,:],scaled_data[training_size:len(scaled_data),:1]


# In[11]:


training_size,test_size


# In[12]:


def dataset_new(dataset, time_step=1):
    dataX,dataY = [] , []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)


# In[13]:


time_step = 20


# In[14]:


x_train, y_train = dataset_new(train_data ,time_step)
x_test, y_test = dataset_new(test_data, time_step)


# In[15]:


print(x_train.shape), print(y_train.shape)


# In[16]:


print(x_test.shape), print(y_test.shape)


# In[17]:


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# Creating The Model

# In[18]:


##creating the model
model=Sequential()
model.add(LSTM(50,return_sequences= True , input_shape=(20,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[19]:


model.summary()


# In[20]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,batch_size=32,verbose=1)


# In[21]:


import tensorflow as tf


# In[22]:


train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[23]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[24]:


from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[25]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[26]:


look_back = 20
trainPredictPlot = np.empty_like(scaled_data)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:] = train_predict

testPredictPlot = np.empty_like(scaled_data)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(scaled_data)-1,:] = test_predict

plt.plot(scaler.inverse_transform(scaled_data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)


# In[27]:


len(test_data)


# In[31]:


x_input = test_data[313:].reshape(1,-1)
x_input.shape


# In[32]:


tem_input=list(x_input)
temp_input=tem_input[0].tolist()


# Predictions

# In[33]:


from numpy import array

first_output=[]
n_steps = 20
i=0
while(i<10):
    if len(temp_input)>20:
        x_input=np.array(temp_input[1:])
        print('{} day output {} '.format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=1)
        print('{} day output {} '.format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        
        first_output.extend(yhat.tolist())
        i=i+1
        
    else:
        x_input = x_input.reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=1)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        first_output.extend(yhat.tolist())
        i = i+1
print(first_output)    


# In[34]:


day_new = np.arange(1,21)
day_predict = np.arange(21,31)


# In[35]:


len(scaled_data)


# In[37]:


plt.plot(day_new,scaler.inverse_transform(scaled_data[2196:]))
plt.plot(day_predict,scaler.inverse_transform(first_output))


# In[39]:


predicted_data=scaled_data.tolist()
predicted_data.extend(first_output)
plt.plot(predicted_data[2200:])


# In[40]:


scaler.inverse_transform(first_output)


# In[ ]:




