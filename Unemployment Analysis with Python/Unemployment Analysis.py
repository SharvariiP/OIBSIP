#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# # Import Data

# In[28]:


df = pd.read_csv(r"C:\Users\ashish\Downloads/Unemployment in India.csv")
df


# In[8]:


unemployed.info()


# In[29]:


df.isnull().sum()


# In[30]:


df.fillna(0, inplace=True)
df.fillna(df.mean(), inplace=True)


# In[31]:


df.isnull().sum()


# In[32]:


df = df.rename(columns={df.columns[0]:'Region',df.columns[3]:'Unemployment_rate',df.columns[4]:'Employed', df.columns[5]:'labour_participation_rate', df.columns[6]:'area'})
df.head()


# In[33]:


df["Region"].unique()


# In[35]:


df2 = pd.read_csv(r"C:\Users\ashish\Downloads/Unemployment_Rate_upto_11_2020.csv")
df2


# In[36]:


df2 = df2.rename(columns={df2.columns[0]:'Region',df2.columns[3]:'Unemployment_rate',df2.columns[4]:'Employed', df2.columns[5]:'labour_participation_rate', df2.columns[6]:'area'})
df2.head()


# In[37]:


heat_maps = df[['Unemployment_rate','Employed','labour_participation_rate']]
heat_maps = heat_maps.corr()
plt.figure(figsize=(12,7))
sns.set_context('notebook',font_scale=1)
sns.heatmap(heat_maps, annot=True,cmap='winter');


# In[38]:


df2.columns= ["Region","Date","Frequency","Unemployment_rate","Employed","labour_participation_rate","area","longitude","latitude"]
plt.figure(figsize=(14, 18))
plt.title("Unemployment_rate")
sns.histplot(x="Unemployment_rate", hue="Region", data=df)
plt.show()


# In[39]:


region = df2.groupby(["Region"])[['Unemployment_rate', "Employed", "labour_participation_rate"]].mean()
region = pd.DataFrame(region).reset_index()
fig = px.bar(region, x="Region", y="Employed", color="Region", title="Average Employed Rate by Region")
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# In[40]:


region = df2.groupby(["Region"])[['Unemployment_rate', "Employed", "labour_participation_rate"]].mean()
region = pd.DataFrame(region).reset_index()
fig = px.bar(region, x="Region", y="Unemployment_rate", color="Region", title="Average Employed Rate by Region")
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# In[43]:


unemployment = df2[["Region", "area", "Unemployment_rate"]]
fig = px.sunburst(unemployment, path=['area','Region'], values='Unemployment_rate', title= 'Unemployment rate in every State and Region', height=700)
fig.show()


# In[ ]:




