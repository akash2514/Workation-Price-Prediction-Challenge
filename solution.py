#!/usr/bin/env python
# coding: utf-8

# In[201]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_absolute_percentage_error
import os
os.chdir(r'C:\Users\akash148363\hackathons\Participant_Data_WPPH\Participant_Data_WPPH')

df = pd.read_csv(r'Train.csv',na_values='Not Available')
print(df.head())
df.dropna(inplace=True)


# In[202]:


df["Travel Date"] = df["Travel Date"].astype('datetime64[ns]')
print(df.isnull().sum())


# In[203]:


df.dtypes


# In[204]:


df_short = df[['Flight Stops','Meals','Per Person Price','Package Type','Destination','Start City','Airline','Itinerary',
              'Sightseeing Places Covered','Hotel Details','Travel Date']]


# In[205]:


df_short.head()


# In[206]:



# Counting the destinations count
# df_short['Destination_count'] = ''
for i in range(len(df_short)):
    df_short.iloc[i,4] = len(str(df_short.iloc[i,4]).split('|'))


# In[207]:


df_short.head() # OK


# In[208]:


# Count of unique airlines
# Counting the destinations count
# df_short['Destination_count'] = ''
# There are not available in airlines so filling na

df_short['Airline'].fillna(11,inplace=True)
for i in range(len(df_short)):
    if df_short.iloc[i,6] != 11:
        df_short.iloc[i,6] = len(str(df_short.iloc[i,6]).split('|'))


# In[209]:


df_short.head() #OK


# In[210]:


# Summing the Itinerary to find no of days to stay
import re
df_short['Itinerary'].fillna(111,inplace=True)
for i in range(len(df_short)):
    StringVar = str(df_short.iloc[i,7])
    if StringVar != 11:
        number=re.findall('\d+',StringVar)
        sum=0
        for j in number:
            sum+=int(j)
            df_short.iloc[i,7] = sum


# In[211]:


df_short.head() #OK


# In[212]:


# Sightseeing Places Covered taking unique count
df_short['Sightseeing Places Covered'].fillna(11,inplace=True)
for i in range(len(df_short)):
    if df_short.iloc[i,8] != 11:
        df_short.iloc[i,8] = len(str(df_short.iloc[i,8]).split('|'))


# In[213]:


df_short.head() #OK


# In[214]:


# Hotel Details to cover the count of distinct hotels
df_short['Hotel Details'].fillna(11,inplace=True)
for i in range(len(df_short)):
    if df_short.iloc[i,9] != 11:
        df_short.iloc[i,9] = len(str(df_short.iloc[i,9]).split('|'))


# In[215]:


df_short.head()


# In[216]:


# get year from the corresponding 
# birth_date column value
df_short['year'] = pd.DatetimeIndex(df['Travel Date']).year
  
# get month from the corresponding 
# birth_date column value
df_short['month'] = pd.DatetimeIndex(df['Travel Date']).month
df_short['day'] = df['Travel Date'].dt.dayofweek
df_short['Weekday_weekend'] = ''

print(list(df_short))



# In[217]:


df_short.head()


# In[218]:



for i in range(len(df_short)):
    if df_short.iloc[i,13] == 6 or df_short.iloc[i,13] == 5:
        df_short.iloc[i,14] = 'Weekday'
    else:
        df_short.iloc[i,14] = 'Weekend'


# In[219]:


df_short.head()


# In[220]:


import string
num2alpha = dict(zip(range(1, 27), string.ascii_lowercase))
num2alpha[2]
col = len(list(df_short))-1
print('col',col)
print(df_short.iloc[i,11])

for i in range(len(df_short)):

    
    if df_short.iloc[i,11]== 2021:
        df_short.iloc[i,11] = 'a'
    elif df_short.iloc[i,11]== 2022:
        df_short.iloc[i,11] = 'b'
    df_short.iloc[i,12] = num2alpha[df_short.iloc[i,12]]


# In[221]:


df_short.head()


# In[222]:


df_short.isnull().sum()


# In[223]:


len(df_short['month'].unique())
print(df_short['month'].unique())
print(df_short['year'].unique())
print(df_short['Weekday_weekend'].unique())


# In[224]:


df_short.head()


# In[225]:


# Dropping unnecessary colums not needed anymore
df_short.drop('day',axis=1,inplace=True) # no need to drop setting the same
df_short.drop('Travel Date',axis=1,inplace=True) # no need to drop setting the same


# In[226]:


# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
df_short['Package Type']= label_encoder.fit_transform(df_short['Package Type']) 
                                                  


# In[227]:


# get dummies
# variables to convert are : Start City
df_short = pd.get_dummies(df_short,drop_first=True,columns=['Start City','Weekday_weekend','year','month'])


# In[228]:


df_short.head()
# df_short.drop('Hotel Details',axis=1,inplace=True)


# In[229]:


# without scaling
X = df_short.drop('Per Person Price',axis=1)
y = df_short['Per Person Price'].values


# In[ ]:





# In[230]:


y


# In[231]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2021)


# In[232]:


# without scaling getting 54% accuracy
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[233]:


clf = RandomForestRegressor(n_estimators=120)
clf.fit(X_train,y_train)


# In[234]:


y_pred = clf.predict(X_test)


# In[235]:


r2_score(y_test,y_pred)


# In[236]:


mean_absolute_error(y_test,y_pred)


# In[237]:


mean_absolute_percentage_error(y_test,y_pred)


# In[238]:


def rmsle(y, y0):
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))


# In[239]:


rmsle(y_test,y_pred)


# In[ ]:


# results with Flight Stops, Meals >> 52.8
# results with Flight Stops, Meals, Package type >> 52.81
# results with Flight Stops, Meals, Package type, Destination count >> 47.17
# results with Flight Stops, Meals, Package type, Destination count, Start City >> 47.17
# results with Flight Stops, Meals, Package type, Destination count, Start City, count of days via Itenary >> 31.84
# results with Flight Stops, Meals, Package type, Destination count, Start City, count of days via Itenary, Sightseeing count >> 29.31
# results with Flight Stops, Meals, Package type, Destination count, Start City, count of days via Itenary, Sightseeing count, Hotel details >> 28.98
# breaking the travel date into year, month ,day has increased the rmsle to 31.22 becasue it should not be label encoded
# try one hot encoding
# 27.93 when left when dropping all NA values


# In[ ]:





# In[ ]:




