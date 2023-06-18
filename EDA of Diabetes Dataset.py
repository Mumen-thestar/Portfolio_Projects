#!/usr/bin/env python
# coding: utf-8

# ### Library import ###

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency


# Reading the file

# In[5]:


df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
df


# In[3]:


#what is in the dataset

df.info()


# Comment:
# No non-null vaues in any column
# Correct data types
# workable column names, so a clean dataset and it is ready for further analysis
# 

# ### Descriptive Statistics

# In[6]:


df.describe()


# In[7]:


#df_std = df.std()
#print(df_std) #entire column
#print(df['HighBP'].std()) #std os a specific column

#df_mean = df.mean()
#df_mean
#print(df['HighBP'].mean()) #mean os a specific column


# Lets create 2 df separating people with diabetes and no diabetes
# 

# ## Exploring the distribution of some columns ##

# The distribution of the Gender throughout the dataset

# In[47]:


df_no = df[df['Diabetes_binary'] == 0.0]
df_yes = df[df['Diabetes_binary'] == 1.0]


# In[ ]:


#countplot with a hue, data = df
#fig = plt.figure(figsize=(15,10))
#ax = sns.countplot(data=df, x='Sex', hue='Diabetes_binary', palette='Set2')


# In[73]:


fig, (ax1, ax2) = plt.subplots(1,2, sharey= True, figsize = (8,4))

ax1 = sns.countplot(data=df_no, x='Sex', ax=ax1, palette='husl')
ax1.set(title='Distribbution of Gender for no_diabetes')
ax1.set_xticklabels(['Female','male'])

ax2 = sns.countplot(data=df_yes, x='Sex', ax=ax2, palette='husl')
ax2.set(title='Distribbution of Gender for diabetes')
ax2.set_xticklabels(['Female','Male']) #Female = 0.0, male = 1.0

plt.show()


# Observation: Both male and female are vulnarable to diabetes

# data distribution of having stroke on the whole population

# In[77]:


fig, (ax1, ax2) = plt.subplots(1,2, sharey= True, figsize= (10,4))

ax1 = sns.countplot(data= df_no, x= 'Stroke', ax=ax1, palette='husl')
ax1.set(title='No Diabetics and Stroke')
ax1.set_xticklabels(['No', 'Yes'])

ax2 = sns.countplot(data= df_yes, x= 'Stroke', ax=ax2, palette='husl')
ax2.set(title='Diabetics and Stroke')
ax2.set_xticklabels(['No', 'Yes'])

plt.show()


# Observation: Individuals with diabetes have an increased likelihood of experiencing strokes.
#     

# The distribution of the Age throughout the dataset

# In[8]:


ax = sns.countplot(data = df_yes, x= 'Age')
ax.set(title='distribution of the Age throughout the diabetic people')
ax.set_xticklabels(['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '>80'], rotation = 50)


# Observation: Diabetes more likely to present among people age 40 and above. most affected by diabetes 60-64, 65-70, 70-74.

# In[37]:


# Check BMI for people with diabetes. 
# We will remove outliers for better visualization, less than 15 and greater than 60.
ax = sns.histplot(data=df_yes, x='BMI')
ax.set(title='BMI distribution for diabetics')
plt.xlim(15, 60)


# 
# People with diabetes have higher BMI.

# In[46]:


# There are some binary columns that we can visualy compare data between no-diabetes and diabetics.
# Lets iterate from those columns and build plots in one go.

col_names = ['HighChol', 'HighBP', 'Smoker', 'HvyAlcoholConsump', 'PhysActivity', 'DiffWalk']
a = 3 #number of rows
b = 2 #number of columns
c = 1 #plot counter

fig = plt.figure(figsize=(12,15))
for i in col_names:
    plt.subplot(a, b, c)
    ax = sns.countplot(data=df, x=i, hue='Diabetes_binary', palette='Set2')
    ax.set(title = '{}'.format(i))
    ax.set(xlabel=None)
    ax.set_xticklabels(['No', 'Yes'])
    ax.legend(['No-diabetes', 'Diabetics'])
    c = c + 1
   


# In[78]:


# Create a correlation matrix
corr_matrix = df.corr()
# Do some conditional formatting for better readability
cm = sns.light_palette("seagreen", as_cmap=True)
corr_matrix_style = corr_matrix.style.background_gradient(cmap = cm)
corr_matrix_style


# In[13]:


# Visualize relationship between all variables
plt.figure(figsize=(12,10))
sns.heatmap(data=corr_matrix, cmap='crest')


# ### Do no-diabetes and diabetics have the same number of poor physical health days per month?

# ### 1. Do no-diabetes and diabetics have the same BMI?
# 
# 

# In[50]:


H0 = 'no-diabetes and diabetics have the same average BMI.'
H1 = 'no-diabetes and diabetics have different average BMI.'

# Creating data groups
df_no_BMI = random.sample(df_no['BMI'])
df_yes_BMI = df_yes['BMI']

# Print ratio of the variance of both data groups
print(np.var(df_no_BMI), np.var(df_yes_BMI))
ratio=  np.var(df_yes_BMI)/ np.var(df_no_BMI)
print(ratio)


# In[16]:


# Check visually how BMI distribution looks like
sns.kdeplot(df_yes_BMI,color='red')
sns.kdeplot(df_no_BMI,color='green')
plt.grid()
plt.title('BMI distribution')
plt.legend(['Diabetics', 'No-diabetes'])


# In[53]:



ttest,p_value_1  = stats.ttest_ind(df_yes_BMI, df_no_BMI)
if p_value_1 < 0.05:   
    print('Reject Null Hypothesis -', H1)
else:
    print('Failed to reject Null Hypothesis -', h0)
print(p_value_1)


# In[ ]:





# In[ ]:




