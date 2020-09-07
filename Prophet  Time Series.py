#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet


# In[3]:


chicago_df_1=pd.read_csv(r'C:\Users\DELL\Desktop\Chicago_Crimes_2005_to_2007.csv',error_bad_lines = False)
chicago_df_2=pd.read_csv(r'C:\Users\DELL\Desktop\Chicago_Crimes_2008_to_2011.csv',error_bad_lines = False)
chicago_df_3=pd.read_csv(r'C:\Users\DELL\Desktop\Chicago_Crimes_2012_to_2017.csv',error_bad_lines = False)


# In[4]:


chicago_df = pd.concat([chicago_df_1,chicago_df_2,chicago_df_3])


# In[5]:


plt.figure(figsize=(10,10))
sns.heatmap(chicago_df.isnull(),cbar=False,cmap='YlGnBu')


# In[6]:


chicago_df.drop(['Unnamed: 0','Case Number','ID','IUCR','X Coordinate','Y Coordinate','Updated On','Year','FBI Code',
                'Beat','Ward','Community Area','Location','District','Latitude','Longitude'],inplace=True,axis=1)


# In[7]:


chicago_df


# In[8]:


chicago_df.Date = pd.to_datetime(chicago_df.Date,format='%m/%d/%Y %I:%M:%S %p')


# In[9]:


chicago_df.index = pd.DatetimeIndex(chicago_df.Date )


# In[10]:


chicago_df


# In[12]:


chicago_df['Primary Type'].value_counts()


# In[13]:


chicago_df['Primary Type'].value_counts().iloc[:15]


# In[37]:


order_data=chicago_df['Primary Type'].value_counts().iloc[:15].index


# In[39]:


plt.figure(figsize=(10,10))
sns.countplot(y='Primary Type',data=chicago_df,order=order_data)


# In[41]:


chicago_df.resample('Y').size()


# In[40]:


plt.plot(chicago_df.resample('Y').size())


# In[42]:


chicago_df


# In[44]:


chicago_prophet=chicago_df.resample('M').size().reset_index()


# In[45]:


chicago_prophet


# In[46]:


chicago_prophet.columns=['Date','Crime Count']


# In[48]:


chicago_prophet


# In[49]:


chicago_prophet_df_final=chicago_prophet.rename(columns={'Date':'ds','Crime Count':'y'})


# In[50]:


chicago_prophet_df_final


# In[52]:


m=Prophet()
m.fit(chicago_prophet_df_final)


# In[55]:


future = m.make_future_dataframe(periods=720)
forecast=m.predict(future)


# In[56]:


m.plot(forecast,)


# In[ ]:




