#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
# convert text into feature vector or numeric values
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[8]:


data = pd.read_csv(r'C:\Users\User\Desktop\spam.csv', encoding='latin')


# In[9]:


data


# In[12]:


data.columns


# In[14]:


data.isnull().sum()


# In[15]:


data.isnull().mean()*100


# In[16]:


data.drop(columns=data[['Unnamed: 2','Unnamed: 3','Unnamed: 4']],axis=1,inplace=True)


# In[18]:


print(data)


# In[19]:


data=data.rename(columns={'v1':'Category','v2':'Message'})
                                    


# In[20]:


data


# In[21]:


data['text length']=data['Message'].apply(len)


# In[22]:


data


# In[32]:


plt.scatter(data= data, x='Category',y='text length')
plt.xlabel("Email Type")
plt.ylabel("text length")
plt.title("scatter plot of Text length with spam/him Differentiation")

plt.show()


# In[42]:


sns.boxplot(data= data, x='Category',y='text length')
plt.xlabel("Email Type")
plt.ylabel("text length")
plt.title("Boxplot of Text length with spam/him Differentiation")

plt.show()


# # label encoding

# In[44]:


#label spam mail as 0; ham mail as 1;

data.loc[data['Category'] == 'spam','Category',] = 0
data.loc[data['Category'] == 'ham','Category',] = 1


# In[45]:


#separating the data as texts and label
X = data['Message']
Y = data['Category']


# In[46]:


print(X)


# In[47]:


print(Y)


# # Splitting the data into training data & test data

# In[48]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)


# In[49]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[50]:


feature_extraction = TfidfVectorizer(min_df = 1,stop_words='english',lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[51]:


Y_train


# In[52]:


print(X_train_features)


# # training the model
# 

# # Logistics regression

# In[54]:


model = LogisticRegression()


# In[55]:


model.fit(X_train_features, Y_train)


# In[57]:


# prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)
     


# In[58]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[59]:


#prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)
     


# In[60]:


print('Accuracy on test data : ', accuracy_on_test_data)


# # building predictive system

# In[61]:


# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data for illustration
input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have b"]

# Sample training data
training_data = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have b",
                 "Get a free gift now! Click here.",
                 "Congratulations, you've won a lottery!",
                 "Invest in this amazing opportunity today.",
                 "Meeting postponed to tomorrow."]

labels = [0, 1, 1, 1, 0]  # 0: Ham mail, 1: Spam mail

# Feature extraction
feature_extraction = CountVectorizer()
input_data_features = feature_extraction.fit_transform(training_data)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(input_data_features, labels)

# Convert the input mail to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# Making prediction
prediction = model.predict(input_data_features)

# Display the prediction
if prediction[0] == 0:
    print('Ham mail')
else:
    print('Spam mail')


# In[ ]:




