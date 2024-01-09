#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px


# In[2]:


data=pd.read_csv(r"C:\Users\User\Desktop\Unemployment in India.csv")
data=pd.read_csv(r"C:\Users\User\Desktop\Unemployment_Rate_upto_11_2020.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


print("Rows--->",data.shape[0])
print("Columns--->",data.shape[1])


# In[6]:


data.isnull()


# In[7]:


data.info()


# In[8]:


data.info


# In[9]:


print(data.describe)


# In[10]:


data.duplicated()


# In[11]:


data.duplicated().sum()


# In[12]:


data.dtypes


# In[13]:


data.columns=['States','Date','Frequency','Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate','Region','longitude','latitude']


# In[14]:


data


# In[15]:


print(data)


# In[16]:


data[['Day','Month','Year']]=data['Date'].str.split('-',expand=True)


# In[17]:


data


# In[18]:


data.drop(columns=['Frequency'],axis=0,inplace=True)
data


# In[19]:


data[:8]


# In[21]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# In[22]:


data.Region.value_counts()


# In[23]:


data.columns
plt.title('Indian Unemployment')
sns.histplot(x="Estimated Employed",hue="Region",data=data)
plt.show()


# In[24]:


plt.figure(figsize=(5,6))
plt.title('Indian Unemployment')
sns.histplot(x="Estimated Unemployment Rate",hue="Region",data=data)
plt.show()


# In[25]:


fg=px.histogram(data,x='States',y='Estimated Unemployment Rate',title='Indian unemployment rate(state-wise)',template='plotly',color='States')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[26]:


fg=px.histogram(data,x='Region',y='Estimated Unemployment Rate',title='Indian unemployment rate(state-wise)',template='plotly',color='Region')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[27]:


data.Month.unique()


# In[28]:


sns.barplot(x="Month",y="Estimated Unemployment Rate",hue="Year",data=data)


# In[29]:


data.Day.unique()


# In[30]:


sns.barplot(x="Day",y="Estimated Unemployment Rate",hue="Year",data=data)


# In[31]:


fg=px.bar(data,x='States',y='Estimated Unemployment Rate',title='Indian unemployment rate(state-wise)',template='plotly',color='States')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[32]:


fg=px.bar(data,x='Region',y='Estimated Unemployment Rate',title='Indian unemployment rate(state-wise)',template='plotly',color='Region')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[33]:


sns.set_theme(style="ticks", palette="pastel")
sns.boxplot(x="Month", y='Estimated Employed', palette=["m", "g"], data=data)
sns.despine(offset=10, trim=True)


# In[47]:


sns.pairplot(data, hue='Region', palette='Dark2')


# In[34]:


data.drop(columns=['Year'],axis=1)


# In[35]:


plt.figure(figsize=[8,8])
plt.title('Indian Unemployment')
sns.boxplot(x="Month",y="Estimated Unemployment Rate",hue="Region",data=data)
plt.show()


# In[36]:


plt.title('Indian Unemployment')
sns.boxplot(x="Day",y="Estimated Unemployment Rate",hue="Region",data=data)
plt.show()


# In[ ]:





# In[37]:


fg=px.scatter(data,x='Region',y='Estimated Unemployment Rate',title='Indian unemployment rate(state-wise)',template='plotly',color='Region')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[38]:


fg=px.scatter(data,x='States',y='Estimated Unemployment Rate',title='Indian unemployment rate(state-wise)',template='plotly',color='States')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[39]:


#Analysing data by sunburst chart
unemploment = data[["Region",'Estimated Unemployment Rate','States']]
figure = px.sunburst(unemploment, path=["Region","States"], 
                     values='Estimated Unemployment Rate',
                     width=700, height=700,  
                     title="Indian unemployment Rate")
figure.show()   


# In[40]:


unemploment = data[["Region",'Estimated Unemployment Rate','States']]
figure = px.sunburst(unemploment, path=["Region","States",'Estimated Unemployment Rate'], 
                     values='Estimated Unemployment Rate',
                     width=1000, height=1000,  
                     title="Indian unemployment Rate")
figure.show()   


# In[61]:


fig =px.line(data, x='Date', y='Estimated Unemployment Rate', title='Unemployment Rate Over the Time', labels={'Estimated Unemployment Rate (%)':'Unemployment Rate'}, template='plotly_dark')
fig.update_traces(line_color='green')

#Covid 19 impact Analysis
covid_events = {'start_date': '2020-03-01', 'end_date': '2021-06-30', 'event': 'Peak of Pandemic'}
fig.add_shape(type='rect', x0=pd.to_datetime(covid_events['start_date']),
              x1=pd.to_datetime(covid_events['end_date']), y0=0, y1=1,
              fillcolor='red', opacity=0.3, layer='below', line=dict(color='red', width=2))
fig.update_layout(xaxis_title='Date', yaxis_title='unemployment Rate', legend_title='Region')
fig.update_xaxes(type='category')
fig.show()


# In[ ]:




