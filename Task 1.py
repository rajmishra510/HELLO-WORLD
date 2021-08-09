#!/usr/bin/env python
# coding: utf-8

# # Rajdeep Mishra

# # Data Science And Business Analytics Intern @ The Sparks Foundation(TSF)

# ## Task 1: Prediction using Supervised Learning

# ## Aim - To predict score of a student if he/she studies for 9.25 hours/day.

# ### Imported all the libraries and packages required for the task

# In[8]:


import pandas as pd  # for manipulating the dataset to profile different datasets
import numpy as np   # for applying numerical operations on the observations
import matplotlib.pyplot as plt  # for plotting the graphs
import seaborn as sns  ## to get more interactive plots
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split     # for splitting the dataset into training and testing sets
from sklearn.linear_model import LinearRegression        # for building the linear regression model
from sklearn.metrics import mean_squared_error,mean_absolute_error  # for calculating mean squared error


# ### Importing the Dataset from Github:

# In[2]:


df_URL='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'


# In[3]:


df=pd.read_csv(df_URL)


# In[5]:


df.head()


# ## Plotting relationship between hours and scores:

# In[19]:


df.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Scores')
plt.xlabel('No. of hours')
plt.ylabel('Score')
plt.show()


# ### From the plot above we can clearly interpret that there is a linear relationship between hours and score. Now we will split our data into two parts: training and testing dataset.

# In[20]:


x=df.iloc[:,0:1]
y=df.iloc[:,1:]


# In[21]:


x


# ### Splitting the dataset into 80:20 ratio into training and testing dataset.

# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# ## Now we will build a Linear Regression Models as follows:

# In[23]:


lr=LinearRegression()


# In[24]:


lr.fit(x_train,y_train)


# In[25]:


lr.score(x_train,y_train)


# In[26]:


lr.score(x_test,y_test)


# In[27]:


pred=lr.predict(x_test)


# ## After the model is trained we need to check its accuracy. For this we will use mean squared error method.

# In[28]:


print(mean_squared_error(pred,y_test))


# In[14]:


print(np.sqrt(mean_squared_error(pred,y_test)))


# ### As we can see that value of MSE is 4.64. Lower the value of MSE, higher the accuracy of the model.

# ## Plotting the best fit line to ascertain the relationships between the points in scatter plot:

# In[29]:


line = lr.coef_*x + lr.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# ## Pedicting the values

# ### Now we can used this trained model to predict the values for test dataset. It will help in ascertaining the accuracy of the model.

# ## Comparing actual values with the predicted values:

# In[16]:


df2=pd.DataFrame(y_test)
df2


# In[17]:


df2['Predicted values']=pred
df2


# ### We know that the MSE is 4.64. The error in the datafram df2 is shown by comparing the actual values against the predicted values.

# ## We have created our Linear Regression Model, with the help of which we will be able to predict the score of a child when the number of studying hours is 9.25/day.

# The Linear Regression Model predictsa numerical variable, when a condition in the numerical variable(s) is given. So now we will set the studying hours to 9.25hrs and predict the values:

# In[18]:


hours= [[9.25]]


# In[19]:


prediction=lr.predict(hours)
prediction


# ## The model is able to predict the score which is 93.691732. We can interpret that if a student studies for 9.25 hours, his/her score will be 93.691732.
