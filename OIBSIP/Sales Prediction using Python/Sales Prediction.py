#!/usr/bin/env python
# coding: utf-8

# In[38]:


pip install -U --pre pycaret


# In[39]:


pip install autoviz


# In[40]:


# Numpy Library for Numerical Calculations
import numpy as np

# Pandas Library for Dataframe
import pandas as pd

# Matplotlib and Seaborn for Plottings
import matplotlib.pyplot as plt
import seaborn as sns

# Pickle Library for Saving the Model
import pickle

# Train_Test_Split for splitting the Dataset
from sklearn.model_selection import train_test_split

# Linear Regression is the Model
from sklearn.linear_model import LinearRegression

# KFold and Cross_Val_Score for Validation
from sklearn.model_selection import cross_val_score

# Metrics is for Analysis of Models
from sklearn import metrics

# Scipy is for Scientific Calculations in Python
from scipy import stats

# Variance Inflation Rate is for getting the change factor in Variance
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Regression is for importing all the regression models
from pycaret.regression import *


# # Import Data

# In[41]:


sales = pd.read_csv(r"C:\Users\ashish\Downloads/Advertising.csv")


# In[42]:


sales.isnull().sum()


# In[43]:


sales.head()


# In[44]:


sales.tail()


# In[45]:


sales.shape


# In[46]:


sales.describe()


# # Visualization

# In[47]:


plt.figure(figsize=(4,4))
sns.scatterplot(data = sales, x = sales['TV'], y = sales['Sales'])
plt.show()


# In[48]:


plt.figure(figsize=(4,4))
sns.scatterplot(data = sales, x = sales['Radio'], y = sales['Sales'])
plt.show()


# In[49]:


plt.figure(figsize=(4,4))
sns.scatterplot(data = sales, x = sales['Newspaper'], y = sales['Sales'])
plt.show()


# # Data Modeling

# In[50]:


X = sales.drop(['Unnamed: 0','Sales'], axis=1)
Y = sales['Sales']
print("X Dimention: ", X.shape)
print("Y Dimention: ", Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=25)


# In[51]:


print("X_Train Shape:", X_train.shape)
print("X_Test Shape:", X_test.shape)
print("Y_Train Shape:", X_train.shape)
print("Y_Test Shape:", Y_test.shape)


# # Model Building

# In[52]:


model = LinearRegression()
model.fit(X_train,Y_train)


# In[53]:


pred = model.predict(X_test)


# # Model Testing

# In[54]:


print('MAE: ', metrics.mean_absolute_error(pred,Y_test))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(pred,Y_test)))
print('R-Squared: ', metrics.r2_score(pred,Y_test))


# # Saving Model

# In[55]:


filename = "Linear_Regression.pkl"
pickle.dump(model, open(filename, 'wb'))
print("Saved the Model")


# # Pycaret
# ##### Comparing Regression Models

# In[56]:


s = setup(data = sales, target = 'Sales', session_id=123)


# In[57]:


compare_models()


# ### Finalizing the Best Model

# In[58]:


etr = create_model('et')


# In[59]:


etr = finalize_model(etr)
etr


# In[60]:


preds = predict_model(etr)
print("Accuracy of the Model is 100%")

