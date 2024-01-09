#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[4]:


sal=pd.read_csv(r"C:\Users\User\Desktop\Advertising.csv")


# In[5]:


sal.head()


# In[6]:


sal


# In[7]:


sal.sample


# In[8]:


sal.isnull()


# In[9]:


sal.isnull().sum()


# In[11]:


sal.columns


# In[12]:


sal.drop(columns=['Unnamed: 0'],axis=1,inplace=True)


# In[13]:


sal


# In[14]:


sal.info()


# In[17]:


sal.describe()


# In[15]:


sal.dtypes


# In[18]:


sal.duplicated()


# In[19]:


sal.duplicated().sum()


# In[34]:


sal.corr()


# In[33]:


plt.figure(figsize=[5,3])
plt.boxplot(sal,vert=False,data=sal,labels=sal.columns, patch_artist=True)
plt.show()


# # Analysing data by histplot graph

# In[29]:


sns.histplot(sal['Newspaper'], color='purple')


# In[30]:


sns.histplot(sal['TV'], color='red')


# In[31]:


sns.histplot(sal['Radio'], color='green')


# # Analyse data by pairplot graph 

# In[26]:


sns.pairplot(sal)


# In[46]:


x=sal.iloc[:,:-1]
x
y = sal.iloc[:,-1:]


# In[38]:


y = sal.iloc[:,-1:]


# In[84]:


xtrain,xtest,ytrain,ytest  = train_test_split(x,y,test_size=0.3,random_state=46)


# In[85]:


xtrain


# In[86]:


xtest


# In[87]:


print(x.shape)
print(xtrain.shape)
print(xtest.shape)


# In[88]:


ytrain,ytest


# # Linear Regression

# In[89]:


lir = LinearRegression()


# In[90]:


lir.fit(xtrain,ytrain)


# In[91]:


ypredict=lir.predict(xtest)
ypredict


# In[92]:


lir.score(xtrain,ytrain)*100


# In[93]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[94]:


xtrain_scaled=sc.fit_transform(xtrain)
xtest_scaled=sc.fit_transform(xtest)


# In[95]:


yprid= lir.predict(xtest_scaled)


# In[96]:


yprid


# In[97]:


plt.scatter(ytest,yprid,c='g')


# In[98]:


mean_squared_error(ytest,yprid)


# In[99]:


mae=mean_absolute_error(ytest,yprid)
mae


# In[100]:


r2_score(ytest,yprid)


# In[101]:


p= np.sqrt(mean_squared_error(ytest,yprid))
p


# In[102]:


mse=mean_absolute_error(ytest,yprid)
mse


# In[103]:


cv = KFold(n_splits=5,shuffle=True, random_state=0)
cv


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




