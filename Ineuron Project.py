#!/usr/bin/env python
# coding: utf-8

# # Hello My Name is RAJDEEP MISHRA

# # Amazon Sales Data Analysis
# 
# ## For this project we will be analysing sales data and finding some trends and relationship between variables we will be also showing some insigths with the help of data visualization so than code and information can be usefull and easy to understand and this data contains features as given below.
# 
# # Data and Setup

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('E:/100 Sales.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# # Exploratory Data Analysis

#  ## Which value of unit cost is highest?

# In[ ]:


df.xs(key='Unit Cost',axis=1).max()


# ## what kind of order priority are there in the data set and there value counts ?

# In[7]:


df['Order Priority'].value_counts().head(5)


#  ## what kind of sale schannel are in the data set and there value counts?

# In[8]:


df['Sales Channel'].value_counts().head(5)


# ## how many types of item are in th data set?

# In[ ]:


df['Item Type'].nunique()


# # Types of items?

# In[10]:


df['Item Type'].unique()


# In[11]:


df['Item Type'].value_counts(dropna=True)


# # Find out the mean value of item type related to the all variables?

# In[12]:


df.groupby('Item Type', as_index=True).max()


# # Data Visualization

# ## Through this plot we can see the no of channels seprately. 

# In[69]:


sns.countplot(x='Sales Channel',data=df,palette='magma')


# ## Through this plot we can see each type of items and their count seprately.

# In[14]:


plt.figure(figsize=(12,3))
sns.countplot(x='Item Type',data=df,palette='viridis')


# ## Through this plot we can see the relation between the 2  variables (units sold) and (sales channel)

# In[15]:


sns.barplot(x='Sales Channel',y='Units Sold',data=df)


# ## Through this plot we can see the relation between the 2 variables (Total profit) and (sales channel)

# In[16]:


sns.barplot(x='Sales Channel',y='Total Profit',data=df,palette='magma',estimator=np.std)
sns.despine()


# In[17]:


sns.boxplot(x="Order Priority", y="Units Sold", data=df,palette='rainbow')


# ## Through this plot we can see the relation between the 2 variables (units sold) and (order priority) but here the priority is magnified by sales channel seprately. 

# In[71]:


plt.figure(figsize=(10,4))
sns.boxplot(x="Order Priority", y="Units Sold",hue='Sales Channel', data=df,palette='rainbow')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)## to relocate legend


# # To see the trends acc. to months, week and hour seprately so we have to create 3 different columns so that we can analyze acc to need.

# In[19]:


type(df['Order Date'].iloc[0])


# In[20]:


df['Order Date'] = pd.to_datetime(df['Order Date'])


# In[21]:


df['Hour'] = df['Order Date'].apply(lambda time: time.hour)
df['Month'] = df['Order Date'].apply(lambda time: time.month)
df['Day of Week'] = df['Order Date'].apply(lambda time: time.dayofweek)


# In[22]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[23]:


df['Day of Week'] = df['Day of Week'].map(dmap)


# In[24]:


df.head()


# In[25]:


plt.figure(figsize=(10,4))
sns.boxplot(x="Day of Week", y="Units Sold",hue='Sales Channel', data=df,palette='rainbow')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)## to relocate legend


# In[39]:


sns.countplot(x='Month',data=df,hue='Sales Channel',palette='rainbow')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[75]:


sns.barplot(x="Month", y="Total Profit", data=df,palette='rainbow')
plt.figure(figsize=(10,4))


# In[49]:


byMonth = df.groupby('Month',as_index=True).mean()
byMonth.head()


# In[50]:


byMonth['Total Revenue'].plot()


# In[51]:


byweek = df.groupby('Day of Week',as_index=True).mean()
byweek.head()


# In[53]:


byweek['Total Revenue'].plot()


# In[26]:


sns.distplot(df['Unit Cost'])


# In[27]:


sns.jointplot(x='Unit Cost',y='Unit Price',data=df,kind='reg')


# In[38]:


df.dropna()


# In[68]:


df.groupby('Country').count()['Total Profit'].plot()
plt.tight_layout()


# # So these are my findings in this data set in which i have shown some relation between variable some trends of sales and other information graphically.

# 
# 
# # THANKYOU!
