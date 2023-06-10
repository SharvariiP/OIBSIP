#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[85]:


# Numpy Library for Numerical Calculations
import numpy as np

# Pandas Library for Dataframe
import pandas as pd

# Matplotlib and for Plottings
import matplotlib.pyplot as plt

# Pickle Library for Saving the Model
import pickle

# RE Library for Regular Expression
import re

# NLTK Library for Natural Language Processing
import nltk
nltk.download('stopwords') # Downloading the Stopwords

# Stopwords for removing stopwords in the Text
from nltk.corpus import stopwords

# PorterStemmer for Stemming the Words
from nltk.stem.porter import PorterStemmer

# CountVectorizer for Bagging of Words and Vectorizing it
from sklearn.feature_extraction.text import CountVectorizer

# Train_Test_Split for splitting the Dataset
from sklearn.model_selection import train_test_split

# Decision Tree Classifier, Random Forest Classifier and Multinomial Naïve Bayes are Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Accuracy Score and Confusion Matrix is for Analysis of Models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# # Import Data

# In[86]:


spam = pd.read_csv(r"C:\Users\ashish\Downloads\spam.csv", encoding='ISO-8859-1')


# In[87]:


spam.isnull().sum()


# In[88]:


spam.head()


# In[89]:


spam.tail()


# In[90]:


spam = spam[['v1', 'v2']]
spam.columns = ['label', 'message']
spam.head()


# In[91]:


spam.shape


# In[92]:


spam.groupby('label').size()


# In[93]:


spam['label'].value_counts().plot(kind='bar')


# # NLP (Natural language processing)
# #### Preprocessing the Text in the Dataset

# In[94]:


ps = PorterStemmer()
corpus = []
for i in range(0, len(spam)):
    review = re.sub('[^a-zA-Z]', ' ', spam['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
# Printing the first 5 values in the corpus list
corpus[1:6]


# ### Creating Bag of Words Model

# In[95]:


cv = CountVectorizer(max_features = 4000)
X = cv.fit_transform(corpus).toarray()
Y = pd.get_dummies(spam['label'])
Y = Y.iloc[:, 1].values


# # Data Modeling
# Splitting the Dataset into Training and Testing Set

# In[96]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=42)


# ### Model Building

# In[97]:


# Model 1 - Random Forest Classifier
model1 = RandomForestClassifier()
model1.fit(X_train, Y_train)

# Model 2 - Decision Tree Classifier
model2 = DecisionTreeClassifier()
model2.fit(X_train, Y_train)

# Model 3 - Multinomial Naïve Bayes
model3 = MultinomialNB()
model3.fit(X_train, Y_train)


# ### Prediction

# In[98]:


pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)


# # Model Testing

# In[99]:


# Model 1 - Random Forest Classifier
print("Random Forest Classifier")
print("Confusion Matrix: ")
print(confusion_matrix(Y_test, pred1))
print("Accuracy: ", accuracy_score(Y_test, pred1))
print("--------------------------------")

# Model 2 - Decision Tree Classifier
print("Decision Tree Classifier")
print("Confusion Matrix: ")
print(confusion_matrix(Y_test, pred2))
print("Accuracy: ", accuracy_score(Y_test, pred2))
print("--------------------------------")

# Model 3 - Multinomial Naïve Bayes
print("Multinomial Naïve Bayes")
print("Confusion Matrix: ")
print(confusion_matrix(Y_test, pred3))
print("Accuracy: ", accuracy_score(Y_test, pred3))


# In[100]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred1)

import seaborn as sns
sns.heatmap(cm, annot=True)


# In[101]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred2)

import seaborn as sns
sns.heatmap(cm, annot=True)


# In[102]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred3)

import seaborn as sns
sns.heatmap(cm, annot=True)


# ## Conclusion - Best Model is Multinomial Naïve Bayes

# ## Saving Models
# #### Saving all the Models

# In[103]:


filename = "RFC.pkl"
pickle.dump(model1, open(filename, 'wb'))
filename = "DTC.pkl"
pickle.dump(model2, open(filename, 'wb'))
filename = "MNB.pkl"
pickle.dump(model3, open(filename, 'wb'))
print("Saved all Models")

