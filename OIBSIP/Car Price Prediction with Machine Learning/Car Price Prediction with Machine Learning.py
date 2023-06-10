#!/usr/bin/env python
# coding: utf-8

# # Importing Library

# In[24]:


# Numpy Library for Numerical Calculations
import numpy as np
# Pandas Library for Dataframe
import pandas as pd
# Math Library for Mathematical Calulations
import math
# Pickle Library for Saving the Model
import pickle
# Matplotlib and Seaborn for Plottings
import matplotlib.pyplot as plt
import seaborn as sns
# Train_Test_Split for splitting the Dataset
from sklearn.model_selection import train_test_split
# Linear Regression, Decision Tree are Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
# Mean Absolute Error, R2 Score and Mean Squared Error is for Analysis of Models
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score


# In[25]:


car_price = pd.read_csv(r"C:\Users\ashish\Downloads\CarPrice.csv")


# In[26]:


car_price.isnull().sum()


# In[27]:


car_price.head()


# In[28]:


car_price.tail()


# In[29]:


car_price.shape


# In[30]:


car_price.describe()


# In[31]:


car_price.groupby('carbody').size()


# In[32]:


car = car_price[["symboling", "wheelbase", "carlength", "carwidth", "carheight", "curbweight", "enginesize", "boreratio", "stroke", "compressionratio", "horsepower", "peakrpm", "citympg", "highwaympg", "price"]]
car


# In[33]:


def plot_bivariate_bar(dataset, cols, width, height, hspace, wspace):
    dataset = dataset.select_dtypes(include = [np.int64])
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = wspace, hspace = hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.int64:
            g = sns.countplot(y = column, data = dataset)
            substrings = [s.get_text()[:15] for s in g.get_yticklabels()]
            g.set(yticklabels = substrings)

plot_bivariate_bar(car, cols = 5, width = 20, height = 15, hspace = 0.2, wspace = 0.5)


# In[34]:


plt.figure(figsize = (10, 10))
sns.jointplot(data = car)
plt.show()


# In[35]:


plt.figure(figsize = (20, 10))
sns.set_style('darkgrid')
sns.heatmap(car.corr(), annot = True, cmap = 'viridis')
plt.show()


# In[36]:


x = car.drop(["price"], 1)
y = car["price"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 16, test_size = 0.25, shuffle=True)


# # Model Building

# In[37]:


model1 = LinearRegression()
model2 = DecisionTreeRegressor()


# In[38]:


model1.fit(xtrain, ytrain)
model1.score(xtrain, ytrain)


# In[39]:


model2.fit(xtrain, ytrain)
model2.score(xtrain, ytrain)


# # Testing Model

# In[40]:


Linear_predictions = model1.predict(xtest)
Decision_predictions = model2.predict(xtest)


# In[41]:


print("Linear Regression Model:")
print("************************")
print('R2_score:', r2_score(ytest, Linear_predictions))
print('Mean Absolute Error:', mean_absolute_error(ytest, Linear_predictions))
print('Mean Squared Error:', mean_squared_error(ytest, Linear_predictions))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(ytest, Linear_predictions)))
print("---------------------------------------------")
print("Decision Tree Regression Model:")
print("******************************")
print('R2_score:', r2_score(ytest, Decision_predictions))
print('Mean Absolute Error:', mean_absolute_error(ytest, Decision_predictions))
print('Mean Squared Error:', mean_squared_error(ytest, Decision_predictions))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(ytest, Decision_predictions)))


# In[42]:


print("Accuracy of Linear Regression Model: ", model1.score(xtrain, ytrain))
print("Accuracy of Decision Tree Regression Model: ", model2.score(xtrain, ytrain))


# # Saving Models

# In[43]:


filename = "Linear_Regression.pkl"
pickle.dump(model1, open(filename, 'wb'))
filename = "Decision_Tree_Regressor.pkl"
pickle.dump(model2, open(filename, 'wb'))
print("Saved all Models")

