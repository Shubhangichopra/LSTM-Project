#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Importing libraries
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pandas_datareader as pdr


# Data Preprocessing

# In[2]:


df= pdr.DataReader('INFY.BO', data_source='yahoo', start='2012-01-01', end='2021-01-01')
df.head()


# In[3]:


##preprocessing data by only using one variable of the data 
## Using only the Close Price Data
df1 = df['Close']
df1.head()


# In[4]:


plt.plot(df1)


# In[5]:


##Rescaling Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(np.array(df1).reshape(-1,1))

scaled_data


# In[6]:


scaled_data.shape


# In[7]:


##spliting the data at 60%
training_size = int(len(scaled_data)*0.60)
test_size = len(scaled_data)-training_size

train_data,test_data = scaled_data[0:training_size,:],scaled_data[training_size:len(scaled_data),:1]


# In[8]:


training_size,test_size


# In[9]:


def dataset_new(dataset, time_step=1):
    dataX,dataY = [] , []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)


# In[10]:


time_step = 50


# In[11]:


x_train, y_train = dataset_new(train_data ,time_step)
x_test, y_test = dataset_new(test_data, time_step)


# In[12]:


print(x_train.shape), print(y_train.shape)


# In[13]:


print(x_test.shape), print(y_test.shape)


# In[14]:


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# Creating the Model

# In[15]:


##creating the model
model=Sequential()
model.add(LSTM(50,return_sequences= True , input_shape=(50,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[16]:


model.summary()


# In[17]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[18]:


import tensorflow as tf


# In[19]:


train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[20]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[21]:


from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[22]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[27]:


look_back = 50
trainPredictPlot = np.empty_like(scaled_data)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:] = train_predict

testPredictPlot = np.empty_like(scaled_data)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(scaled_data)-1,:] = test_predict

plt.plot(scaler.inverse_transform(scaled_data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)


# In[43]:


len(test_data)


# In[44]:


x_input = test_data[837:].reshape(1,-1)
x_input.shape


# In[45]:


tem_input=list(x_input)
temp_input=tem_input[0].tolist()


# Predictions

# In[46]:


from numpy import array

first_output=[]
n_steps = 50
i=0
while(i<10):
    if len(temp_input)>50:
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


# In[47]:


day_new = np.arange(1,51)
day_predict = np.arange(51,61)


# In[48]:


len(scaled_data)


# In[49]:


plt.plot(day_new,scaler.inverse_transform(scaled_data[2166:]))
plt.plot(day_predict,scaler.inverse_transform(first_output))


# In[50]:


predicted_data=scaled_data.tolist()
predicted_data.extend(first_output)
plt.plot(predicted_data[2000:])


# In[51]:


scaler.inverse_transform(first_output)


# In[52]:


from numpy import array

first_output=[]
n_steps = 50
i=0
while(i<100):
    if len(temp_input)>50:
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


# In[53]:


day_new = np.arange(1,51)
day_predict = np.arange(51,151)


# In[54]:


plt.plot(day_new,scaler.inverse_transform(scaled_data[2166:]))
plt.plot(day_predict,scaler.inverse_transform(first_output))


# In[ ]:




