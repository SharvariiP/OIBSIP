#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# # Import Data set

# In[3]:


iris = pd.read_csv(r"C:\Users\ashish\Downloads/Iris.csv")


# In[4]:


iris.isnull().sum()


# In[5]:


iris.head()


# In[6]:


iris.tail()


# In[7]:


iris = iris.drop(['Id'], axis=1)
iris.columns


# In[8]:


iris.shape


# In[9]:


print(iris.describe())


# In[10]:


print(iris.groupby('Species').size())


# # Visualization of the Data

# In[11]:


species_plot = iris['Species'].value_counts().plot.bar(title = 'Flower Class Distribution')
species_plot.set_xlabel('Class',size=20)
species_plot.set_ylabel('Count',size=20)


# In[12]:


iris.plot(kind = 'box', subplots = True, layout = (2, 2), sharex = False, sharey = False, title = "Box and Whisker plot for each Attribute")
plt.show()


# In[13]:


iris.hist()
plt.show()


# In[14]:


sns.set(style = "ticks")
sns.pairplot(iris, hue = "Species")


# ## Data Modeling

# In[15]:


X = iris.drop(['Species'], axis=1)
Y = iris['Species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=25)


# In[16]:


print("X_Train Shape:", X_train.shape)
print("X_Test Shape:", X_test.shape)
print("Y_Train Shape:", X_train.shape)
print("Y_Test Shape:", Y_test.shape)


# ## Model Building

# In[17]:


# All the Models will be stored in this models list.
models = []
# Linear Models
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = "auto")))
models.append(('LDA', LinearDiscriminantAnalysis()))
# Non-linear Models
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVC', SVC(gamma = "auto")))
print("Model Accuracy")
# Evaluating each Models
names = []
accuracy = []
for name, model in models: 
    # 15 Cross Fold Validation for each Models
    kfold = KFold(n_splits=15)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    # Displaying the Accuracy of each Model in Validation
    names.append(name)
    accuracy.append(cv_results)
    msg = "%s: Accuracy = %f" % (name, cv_results.mean())
    print(msg)


# ## Testing Model

# In[18]:


models = []
# Linear Models
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = "auto")))
models.append(('LDA', LinearDiscriminantAnalysis()))
# Non-linear Models
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVC', SVC(gamma = "auto")))


# In[19]:


def test_model(model):
    # Training the Dataset with Training Set
    model.fit(X_train, Y_train)
    # Predicting the Values with Testing Set
    predictions = model.predict(X_test)
    # Model Testing Results
    print("Accuracy:", accuracy_score(Y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, predictions))
    print("Classification Report:")
    print(classification_report(Y_test, predictions))
# Predicting the Values
for name, model in models:
    print("----------------")
    print("Testing:", name)
    test_model(model)


# # Saving Models

# In[20]:


for name, model in models:
    filename = name + ".pkl"
    pickle.dump(model, open(filename, 'wb'))
print("Saved all Models")

