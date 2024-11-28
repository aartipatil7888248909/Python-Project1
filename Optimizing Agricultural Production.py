#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact


# In[2]:


data = pd.read_csv('C:\\Users\\aarti\\OneDrive\\Documents\\My Achievement\\Python project1\\data.csv')


# In[3]:


print("Shape of the Dataset :",data.shape)


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[6]:


data['label'].value_counts()


# In[7]:


print("Average Ratio of Nitrogen in the soil : (0:2f)".format(data['N'].mean()))
print("Average Ratio of Phosphorous in the soil : (0:2f)".format(data['P'].mean()))
print("Average Ratio of Potassium in the soil : (0:2f)".format(data['K'].mean()))
print("Average Ratio of Celsius in the soil : (0:2f)".format(data['temperature'].mean()))
      
print("Average Relative Humidity in %  : (0:2f)".format(data['humidity'].mean()))
print("Average PH value of the soil : (0:2f)".format(data['ph'].mean()))
print("Average Rainfall in mm soil : (0:2f)".format(data['rainfall'].mean()))


# In[8]:


@interact
def summary(crops = list(data['label'].value_counts().index)):
    x = data[data['label']==crops]
    print("____________________________________")
    print("Statics for Nitrogen")
    print("Minimum Nirogen required :",x['N'].min())
    print("Average Nirogen required :",x['N'].mean())
    print("maximum required :",x['N'].max())
    print("____________________________________")
    
    print("Statics for Phosphorous")
    print("Minimum Phosphorous required :",x['P'].min())
    print("Average Phosphorous required :",x['P'].mean())
    print("maximum Phosphorous required :",x['P'].max())
    
    print("____________________________________")
    print("Statics for Potassium")
    print("Minimum Potassium  required :",x['K'].min())
    print("Average Potassium  required :",x['K'].mean())
    print("maximum Potassium  required :",x['K'].max())
    
    print("____________________________________")
    print("Statics for Temperature")
    print("Minimum Temperture required : {0:2f}".format(x['temperature'].min()))
    print("Average Temperture required : {0:2f}".format(x['temperature'].mean()))
    print("Maximum Temperture required : {0:2f}".format(x['temperature'].max()))
    
    print("____________________________________")
    print("Statics for Humidity")
    print("Minimum Humidity required : {0:2f}".format(x['humidity'].min()))
    print("Average Humidity required : {0:2f}".format(x['humidity'].mean()))
    print("Maximum Humidityrequired : {0:2f}".format(x['humidity'].max()))
    
    
    print("____________________________________")
    print("Statics for PH")
    print("Minimum Humidity required : {0:2f}".format(x['ph'].min()))
    print("Average Humidity required : {0:2f}".format(x['ph'].mean()))
    print("Maximum Humidity required : {0:2f}".format(x['ph'].max()))
    
    print("____________________________________")
    print("Statics for Rainfall")
    print("Minimum Rainfall  required : {0:2f}".format(x['rainfall'].min()))
    print("Average Rainfall  required : {0:2f}".format(x['rainfall'].mean()))
    print("Maximum Rainfall required : {0:2f}".format(x['rainfall'].max()))
    
    

    
     


# In[9]:


@interact
def compare(Conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Average Value for", Conditions, "is {0:.2f}".format(data[Conditions].mean()))
    print("___________________________________")
    print("Rice : {0:.2f}".format(data[(data['label'] == 'rice')][Conditions].mean()))
    print("Black Frams : {0:.2f}".format(data[(data['label'] == 'blackgram')][Conditions].mean()))
    print("Banana : {0:.2f}".format(data[(data['label'] == 'banana')][Conditions].mean()))
    print("Jute : {0:.2f}".format(data[(data['label'] == 'jute')][Conditions].mean()))
    print("Coconut : {0:.2f}".format(data[(data['label'] == 'coconut')][Conditions].mean()))
    print("Apple : {0:.2f}".format(data[(data['label'] == 'apple')][Conditions].mean()))
    print("Papaya : {0:.2f}".format(data[(data['label'] == 'papaya')][Conditions].mean()))
    print("Muskmelon : {0:.2f}".format(data[(data['label'] == 'muskmelon')][Conditions].mean()))
    print("Grapes : {0:.2f}".format(data[(data['label'] == 'grapes')][Conditions].mean()))
    print("Watermelon : {0:.2f}".format(data[(data['label'] == 'watermelon')][Conditions].mean()))
    print("Kidney beans : {0:.2f}".format(data[(data['label'] == 'kidney beans')][Conditions].mean()))
    print("Mung Beans : {0:.2f}".format(data[(data['label'] == 'mung Beans')][Conditions].mean()))
    print("Oranges : {0:.2f}".format(data[(data['label'] == 'oranges')][Conditions].mean()))
    print("Chick Peas : {0:.2f}".format(data[(data['label'] == 'chick peas')][Conditions].mean()))
    print("Lentils : {0:.2f}".format(data[(data['label'] == 'lentils')][Conditions].mean()))
    print("Cotton : {0:.2f}".format(data[(data['label'] == 'cotton')][Conditions].mean()))
    print("Maize : {0:.2f}".format(data[(data['label'] == 'maize')][Conditions].mean()))
    print("Moth Beans : {0:.2f}".format(data[(data['label'] == 'moth beans')][Conditions].mean()))
    print("Pigeon Peas : {0:.2f}".format(data[(data['label'] == 'pigeon peas')][Conditions].mean()))
    print("Mango : {0:.2f}".format(data[(data['label'] == 'mango ')][Conditions].mean()))
    print("Pomegranted : {0:.2f}".format(data[(data['label'] == 'pomegranted')][Conditions].mean()))
    print("Coffee : {0:.2f}".format(data[(data['label'] == 'coffee ')][Conditions].mean()))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


# In[10]:


@interact
def compare(Conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Crops which require greater than average", Conditions,'\n')
    print(data[data[Conditions] > data[Conditions].mean()]['label'].unique())
    print("------------------------------------------------------------")
    
    print("Crops which require less than average", Conditions,'\n')
    print(data[data[Conditions] <= data[Conditions].mean()]['label'].unique())
   
    
    


# # Distribution

# In[52]:


plt.rcParams['figure.figsize'] = (15,7)

# First subplot
plt.subplot(2, 4, 1)
sns.distplot(data['N'], color='lightgrey')
plt.xlabel('Ratio of Nitrogen', fontsize=12)
plt.grid()

# Second subplot
plt.subplot(2, 4, 2)
sns.distplot(data['P'], color='skyblue')
plt.xlabel('Ratio of Phosphorus', fontsize=12)  # Fixed spelling here
plt.grid()

# Third subplot
plt.subplot(2, 4, 3)
sns.distplot(data['K'], color='lightblue')
plt.xlabel('Ratio of Potassium', fontsize=12)
plt.grid()

# Fourth subplot
plt.subplot(2, 4, 4)
sns.distplot(data['temperature'], color='red')
plt.xlabel('Ratio of Temperature', fontsize=12)
plt.grid()

# Fifth subplot
plt.subplot(2, 4, 5)
sns.distplot(data['rainfall'], color='green')
plt.xlabel('Ratio of Rainfall', fontsize=12)
plt.grid()

# Sixth subplot
plt.subplot(2, 4, 6)
sns.distplot(data['humidity'], color='lightpink')  # Fixed color spelling here
plt.xlabel('Ratio of Humidity', fontsize=12)
plt.grid()

# Seventh subplot
plt.subplot(2, 4, 7)
sns.distplot(data['ph'], color='lightgreen')
plt.xlabel('Ratio of PH level', fontsize=12)
plt.grid()


# In[ ]:


plt.rcParams['figure.figsize'] = (15,7)

# First subplot
plt.subplot(2, 4, 1)
sns.distplot(data['N'], color='lightgrey')
plt.xlabel('Ratio of Nitrogen', fontsize=12)
plt.grid()

# Second subplot
plt.subplot(2, 4, 2)
sns.distplot(data['P'], color='skyblue')
plt.xlabel('Ratio of Phosphorus', fontsize=12)  # Fixed spelling here
plt.grid()

# Third subplot
plt.subplot(2, 4, 3)
sns.distplot(data['K'], color='lightblue')
plt.xlabel('Ratio of Potassium', fontsize=12)
plt.grid()

# Fourth subplot
plt.subplot(2, 4, 4)
sns.distplot(data['temperature'], color='red')
plt.xlabel('Ratio of Temperature', fontsize=12)
plt.grid()

# Fifth subplot
plt.subplot(2, 4, 5)
sns.distplot(data['rainfall'], color='green')
plt.xlabel('Ratio of Rainfall', fontsize=12)
plt.grid()

# Sixth subplot
plt.subplot(2, 4, 6)
sns.distplot(data['humidity'], color='lightpink')  # Fixed color spelling here
plt.xlabel('Ratio of Humidity', fontsize=12)
plt.grid()

# Seventh subplot
plt.subplot(2, 4, 7)
sns.distplot(data['ph'], color='lightgreen')
plt.xlabel('Ratio of PH level', fontsize=12)
plt.grid()


# In[12]:


#lets find out some interesting fact
print("Some Interesting Patterns")
print("-----------------------")
print("Crops which requires very high rate of Nitrogen content in soil:",data[data['N'] > 120]['label'].unique())
print("Crops which requires very high rate of Phosphorous content in soil:",data[data['P'] > 100]['label'].unique())
print("Crops which requires very high rate of Potassium content in soil:",data[data['K'] > 200]['label'].unique())
print("Crops which requires very high rate of Rainfall content in soil:",data[data['rainfall'] < 10]['label'].unique())
print("Crops which requires very high rate of Temperature content in soil:",data[data['temperature'] < 120]['label'].unique())
print("Crops which requires very high rate of Temperature content in soil:",data[data['temperature'] > 120]['label'].unique())
print("Crops which requires very high rate of Low Humidity content in soil:",data[data['N'] < 10]['label'].unique())
print("Crops which requires very high rate of Low PH content in soil:",data[data['ph'] > 40]['label'].unique())
print("Crops which requires very high rate of High PH content in soil:",data[data['ph'] > 120]['label'].unique())


# In[16]:


#check Season 
print("Summer Crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("-------------------------------------------------------------------------")
print("Winter Crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print("-------------------------------------------------------------------------")
print("Rainy Crops")
print(data[(data['temperature'] < 200) & (data['humidity'] > 30)]['label'].unique())


# In[20]:


from sklearn.cluster import KMeans  # Corrected import
import warnings
import pandas as pd
x = data.loc[:,['N','P','K','temperature','ph','humidity','rainfall']].values
print(x.shape)
x_data  = pd.DataFrame(x)
x_data.head()


# In[ ]:





# In[ ]:





# In[21]:


#split the dataset for predictive modelling
y = data['label']
x = data.drop(['label'], axis = 1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[22]:


#Training and Testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
print("The shape of x train:",x_train.shape)
print("The shape of x test:",x_test.shape)
print("The shape of y tain:",y_train.shape)
print("The shape of y test:",x_test.shape)


# In[39]:


# crete a prediction model

from sklearn.linear_model import LogisticRegression 
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

y_pred


# In[40]:


from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)


# In[41]:


data.head()


# In[42]:


prediction = model.predict((np.array([[90,
                                      40,
                                      40,
                                      20,
                                      80,
                                      7,
                                      200]])))
print("The Suggested crop for given climatic condition is :",prediction)


# In[ ]:





# In[43]:


#check orang data


# In[46]:


data[data['label'] == 'orange'].head()


# In[49]:


prediction = model.predict((np.array([[20,
                                      30,
                                      10,
                                      15,
                                      90,
                                      7.5,
                                      100]])))
print("The Suggested crop for given climatic condition is :",prediction)


# In[50]:


data[data['label'] == 'coconut'].head()


# In[ ]:




