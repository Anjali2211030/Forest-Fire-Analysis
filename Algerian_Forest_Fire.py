#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


# Importing the Dataset
df = pd.read_csv('D:3rd Semester/Algerian_data_2.csv')


# In[3]:


# Copying to the dataframe
df = df.copy()


# In[ ]:


#Need to understand the dataset
#1


# In[4]:


# Display First 5 rows of dataset
df.head()


# In[5]:


# Display Last 5 Rows of Dataset
df.tail()


# In[6]:


# Know the datatypes
df.dtypes


# In[7]:


df.info()


# In[9]:


# Display the shape of the shape of the dataset : Number of rows and the Number of columns
df.shape


# In[10]:


# Getting overall statistics about the dataframe
df.describe(include='all')


# In[11]:


# How many columns are in the dataframe
df.columns


# In[13]:


# Remaining column names so as to remove spaces Behind

df.rename(columns={'Rain' : 'Rain', 'Classes  ': 'classes'}, inplace=True)


# In[14]:


df.nunique()


# In[15]:


# Cleaning the data
# Checking for null values and dropping them if any
df.isnull().sum()


# In[ ]:


# NO null values


# In[17]:


# Checking after dropping year attribute
df = df.drop(['year', 'classes'], axis=1)


# In[18]:


df.head()


# In[20]:


# Relationship Analysis
# A heatmap is a graphical representation of data that uses a system of colour coding to represent different values
# using a correaltion heatmap to view relationship between variables.
sns.heatmap(df.corr(),annot=True,cmap='viridis',linewidths=.5)


# In[21]:


# Histogram
df.hist(figsize=(20,14),color='b')


# In[24]:


# Lineplot
sns.lineplot(x='Temperature',y='Day',data=df,color='g')


# In[29]:


# Pairplot
sns.pairplot(df)


# In[31]:


#JointPlot
sns.jointplot(x='Month',y='Temperature',data=df,color='r')


# In[32]:


# Barplot
plt.style.use("default")
sns.barplot(x="Day", y="Temperature",data=df)
plt.title("Day vs Temperature", fontsize=15)
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.show()


# In[37]:


df.groupby(['year-month']).Classes.value_counts().unstack('Classes').plot.bar(title = 'Month-Year Wise Classes')
plt.show()


# In[38]:


df.plot.box(grid='True')


# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[75]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_ptrdictions = logreg.predict(X_test)


# In[46]:


X = df[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DC', 'ISI']]
Y = df["Classes"]


# In[62]:


X = df[['Temperature','RH', 'Ws', 'Rain', 'FFMC', 'DC', 'ISI']]
X['intercept'] = 1


# In[48]:


df.columns


# In[49]:


X = df[['Temperature','RH', 'Ws', 'Rain', 'FFMC', 'DC', 'ISI']]
X['intercept'] = 1 


# In[60]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[['Temperature','RH', 'Ws', 'Rain', 'FFMC', 'DC', 'ISI']]
X['intercept'] = 1 

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif


# In[53]:


list1 = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

for i in list1:
    df[i] = pd.to_numeric(df[i], errors='coerce')


# In[63]:


import seaborn as sns
print(df['Day'].value_counts())
sns.barplot(x=df.Day.value_counts().index, y=df.Day.value_counts(), order=['Sun', 'Mon', 'Tue', 'Wed', 'Thurs', 'Fri','Sat'])
sns.set_xlabel("Day of the Week")
sns.set_ylabel("Number of Fires")
sns.set_title("Occurances of Fires by Weekday")


# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# In[65]:


df = ["This is a positive example", "Negative sentiment here", "Another positive instance", "Negative review"]
df = ["Positive", "Negative", "Positive", "Negative"]


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(df, df, test_size=0.25, random_state=42)


# In[68]:


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# In[69]:


classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)


# In[70]:


predictions = classifier.predict(X_test_vec)


# In[71]:


accuracy = metrics.accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")


# In[72]:


conf_matrix = metrics.confusion_matrix(y_test, predictions)
classification_report = metrics.classification_report(y_test, predictions)


# In[73]:


print("\nConfusion Matrix:")
print(conf_matrix)


# In[74]:


print("\nClassification Report:")
print(classification_report)


# In[79]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_vec,y_train)
y_predictions = logreg.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predictions))


# In[80]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Assuming you have X_train_vec, X_test_vec, y_train, and y_test from previous code

# Create and train the Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train_vec, y_train)

# Make predictions on the test set
y_predictions = logreg.predict(X_test_vec)

# Print the classification report
print(classification_report(y_test, y_predictions))


# In[81]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Sample dataset (replace this with your own dataset)
data = ["This is a positive example", "Negative sentiment here", "Another positive instance", "Negative review"]
labels = ["Fire", "Not Fire", "Fire", "Not Fire"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

# Create a CountVectorizer to convert text data into a bag-of-words representation
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Make predictions on the test set
y_predictions = nb_classifier.predict(X_test_vec)

# Print the classification report
report = classification_report(y_test, y_predictions)
print(report)


# In[82]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Sample dataset (replace this with your own dataset)
df = ["This is a positive example", "Negative sentiment here", "Another positive instance", "Negative review"]
Classes = ["Fire", "Not Fire", "Fire", "Not Fire"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, Classes, test_size=0.25, random_state=42)

# Create a CountVectorizer to convert text data into a bag-of-words representation
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Make predictions on the test set
y_predictions = nb_classifier.predict(X_test_vec)

# Evaluate the performance
accuracy = metrics.accuracy_score(y_test, y_predictions)
print(f"Accuracy: {accuracy:.2f}")

# Print confusion matrix and classification report
conf_matrix = metrics.confusion_matrix(y_test, y_predictions)
classification_report = metrics.classification_report(y_test, y_predictions)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.datasets import load_iris

# Load a sample dataset (replace this with your own dataset)
df = load_df()
X = iris.data
y = Classes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_predictions = dt_classifier.predict(X_test)

# Evaluate the performance
accuracy = metrics.accuracy_score(y_test, y_predictions)
print(f"Accuracy: {accuracy:.2f}")

# Print confusion matrix and classification report
conf_matrix = metrics.confusion_matrix(y_test, y_predictions)
classification_report = metrics.classification_report(y_test, y_predictions)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report)


# In[4]:


log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.4f}'
     .format(log_reg.score(X_train_scaled, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.4f}'
     .format(log_reg.score(X_test_scaled, y_test)))
     


# In[5]:


plt.style.use('ggplot')
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=importance_df, x='importance', y='feature',ec = 'black')
ax.set_title('Top 5 Important Features', weight='bold',fontsize = 15)
ax.set_xlabel('Feature Importance %',weight='bold')
ax.set_ylabel('Features',weight='bold')
plt.show()


# In[ ]:




