#!/usr/bin/env python
# coding: utf-8

# # RAJDEEP MISHRA

# # Data Science and Business Analytics Intern @ The Sparks Foundation(TSF)

# ## Task 5: Exploratory Data Analysis - Sports

# ## Aim-  Find out the most successful teams, players and factors contributing win or loss of a team.

# ### Importing all libraries and packages required.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ### Reading the dataset 

# In[2]:


matches=pd.read_csv("D:\matches.csv")
matches.head(10)


# In[3]:


deliveries=pd.read_csv("D:\deliveries.csv")
deliveries.head(10)


# ### Getting info of both the datasets

# In[4]:


matches.info()


# In[5]:


deliveries.info()


# ### Displaying the number of rows and columns

# In[6]:


matches.shape


# In[7]:


deliveries.shape


# ### Checking if any null value is present 

# In[8]:


matches.isnull().sum()


# In[9]:


deliveries.isnull().sum()


# ### Removing the null values 

# In[10]:


matches.dropna(axis=1,how="all",inplace=True)


# ### Getting Statistical summary of the datasets 

# In[11]:


matches.describe()


# In[12]:


deliveries.describe()


# In[13]:


print('Total Matches Played is :',matches.shape[0])
print('Total Seasons Played is :',len(matches["season"].unique()))


# ### Amalysis Of Matches Dataset 

# ### We are conactenating all the unique teams.

# In[14]:


pd.concat((matches['team1'],matches['team2'])).unique()


# ### Season with most number of matches 

# In[15]:


matches['season'].value_counts()


# ### Graph Showing how many matches are played in season year

# In[16]:


plt.figure(figsize=(12,6))
sea.countplot('season',data=matches,palette='bright')
plt.xlabel('Season Year')
plt.ylabel('Matches Players')
plt.show()


# ### Graph showing matches won by each team

# In[17]:


plt.figure(figsize=(12,6))
sea.countplot('winner',data=matches,palette='bright')
plt.title('Matches won by each team')
plt.xticks(rotation=90)
plt.show()


# ### List of top 10 most successful players during all seaons

# In[18]:


matches['player_of_match'].value_counts()[:10]


# In[19]:


names=matches["player_of_match"].value_counts()[:10].index
values=matches["player_of_match"].value_counts()[:10].values


# ### Graph showing Man of the Match player

# In[20]:


plt.figure(figsize=(12,6))
sea.barplot(names,values,palette='bright')
plt.title('Man of the match')
plt.xlabel('Player name')
plt.ylabel('Matches')
plt.xticks(rotation=90)
plt.show()


# In[21]:


matches.columns


# ### Graph showing Toss Decisions

# In[22]:


plt.figure(figsize=(12,6))
sea.countplot(matches["season"].sort_values(),hue=matches['toss_decision'],palette='bright')
plt.title('Toss decision')
plt.show()


# ### Graph representing Matches And Venues

# In[24]:


names=matches["venue"].value_counts()[:10].index
values=matches["venue"].value_counts()[:10].values

plt.figure(figsize=(12,6))
sea.barplot(values,names,palette='bright')
plt.show()


# ### Graph showing toss won by each team

# In[25]:


plt.figure(figsize=(12,6))
sea.countplot(x='toss_winner',data=matches,palette='bright')
plt.title('Toss won by each team')
plt.xticks(rotation=90)
plt.show()


# ### Pie Chart representing teams that won both matches and toss

# In[26]:


wins=matches['toss_winner']==matches["winner"]
wins.value_counts().plot(kind='pie')
plt.show()


# ### Chart showing teams that choose bat first and won by runs

# In[27]:


import plotly.express as px

labels=matches[matches["win_by_runs"]!=0]['winner'].value_counts()[:10].index
values=matches[matches["win_by_runs"]!=0]['winner'].value_counts()[:10].values
fig=px.pie(matches,names=labels,values=values,title='Pie Chart',hole=.5,hover_name=labels)
fig.show()


# ## Analysis of Deliveries dataset

# ### Total extra runs during all seasons

# In[28]:


deliveries['extra_runs'].sum()


# ### Total noball runs during all seasons

# In[29]:


deliveries['noball_runs'].sum()


# ### Total wide runs during all seasons

# In[30]:


deliveries['wide_runs'].sum()


# ### Total penalty runs during all seasons

# In[31]:


deliveries['penalty_runs'].sum()


# ### Graph showing player out and count

# In[32]:


plt.figure(figsize=(12,6))
sea.countplot(x='dismissal_kind',data=deliveries,palette='bright')
plt.title('Outs And Counts')
plt.xticks(rotation=90)
plt.show()


# ## Most Successful teams

# ### Mumbai Indians

# ### Chennai Super Kings

# ### Kolkata Knight Riders

# ## Most Successful Players

# ### Chris Gayle

# ### AB de Villers

# ## The factors contributing win or loss of a team depends on Toss Win, Choose bat first or not, Performance of team, etc
