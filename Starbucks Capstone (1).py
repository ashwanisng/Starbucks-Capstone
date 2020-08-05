#!/usr/bin/env python
# coding: utf-8

# <!-- ### Table of content 
# 
# 1- [Project Definition](#Definition)<br>
# 1.1- [Project Overview](#Overview)<br>
# 1.2- [Problem Statement](#Statement)<br> 
# 1.3- [Metrics](#Metrics)<br>
# 2- [Exploratory Data Analysis](#EDA)<br>
# 2.1- [Data Exploration and Visualization:](#Exploration)<br> 
# 2.1.A- [Portfolio Dataset](#Portfolio1)<br>
# 2.1.B- [Profile Dataset](#Profile1)<br> 
# 2.1.C- [Transcript Dataset](#Transcript1)<br> 
# 2.2- [Data Analysis](#Anaysis1)<br> 
# 3- [Data Preprocessing (Wrangling/Cleaning):](#Preprocessing)<br>
# 3.A- [Portfolio Dataset](#Portfolio2)<br>
# 3.B- [Profile Dataset](#Profile2)<br>
# 3.C-[Transcript Dataset](#Transcript2)<br>
# 4- [Data Modeling:](#Modeling)<br>
# 4.1- [Modeling](#Modeling)<br>
# 4.2- [Model Evaluation](#Evaluation)<br>
# 4.3- [Model Refinement](#Refinement)<br>
# 5- [Conclusion](#Conclusion)<br>
# 6- [Reflection](#Reflection)<br>
# 7- [Improvement](#Improvement)<br> -->

# # Starbucks Capstone Challenge
# 
# ### Introduction
# 
# This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 
# 
# Not all users receive the same offer, and that is the challenge to solve with this data set.
# 
# Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.
# 
# You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 
# 
# Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.
# 
# ### Example
# 
# To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
# 
# However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.
# 
# ### Cleaning
# 
# This makes data cleaning especially important and tricky.
# 
# You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.
# 
# ### Final Advice
# 
# Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# ### 1.1.2) Project's Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record
# 
# **Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  
# 
# You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:
# 
# <img src="pic1.png"/>
# 
# Then you will want to run the above command:
# 
# <img src="pic2.png"/>
# 
# Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.

# In[1]:


import pandas as pd
import numpy as np
import math
import json
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'notebook')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor


# In[2]:


# read in the json files
portfolio = pd.read_json('Data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('Data/profile.json', orient='records', lines=True)
transcript = pd.read_json('Data/transcript.json', orient='records', lines=True)


# #  Portfolio Dataset

# In[3]:


portfolio.head()


# - Portfolio dataset contain -
# <ol>
#     <li> reward </li>
#     <li> channels </li>
#     <li> difficulty </li>
#     <li> duration </li>
#     <li> offer_type </li>
#     <li> id </li>
# 
# 
# </ol>

# In[4]:


# data info

portfolio.info()


# In[5]:


# describe the dataset

profile.describe()


# In[6]:


# Number of row in the dataset 

print("Number of row in the dataset are : ", portfolio.shape[0])


# In[7]:


# Number of column in the dataset 

print("Number of column in the dataset are : ", portfolio.shape[1])


# In[8]:


# check weather there is a null value or not

portfolio.isnull().sum()


# **There are no null value in the dataset.**

# In[9]:


# checking the offer types the customer can receive
portfolio['offer_type'].unique()


# In[10]:


# check for how type pf offer there are

portfolio['offer_type'].value_counts()


# **There are 3 type offers**
# - Boggo
# - Discount
# - informational

# In[11]:


# datatype of dataframe

portfolio.dtypes


# ## Profile DataSet

# In[12]:


profile.head()


# - Profile dataset contain -
# <ol>
#     <li> Gender </li>
#     <li> Age </li>
#     <li> Id </li>
#     <li> Became_member_on </li>
#     <li> Income </li>
# 
# 
# </ol>

# In[13]:


# Number of row in the dataset 

print("Number of row in the dataset are : ", profile.shape[0])


# In[14]:


# Number of column in the dataset 

print("Number of column in the dataset are : ", profile.shape[1])


# In[15]:


# data info

profile.info()


# In[16]:


# check weather there is a null value or not

profile.isnull().sum()


# In[17]:


# describe the dataset

profile.describe()


# In[18]:


# datatype of dataframe

profile.dtypes


# In[20]:


# check for age

profile.sort_values(by='age', ascending=False)


# **How age can be 118......????????**
# 

# ### Let's analyis for 118 year people.

# In[21]:


profile[profile['age']==118].count()


# In[22]:


profile[['gender', "age", "income"]][profile['age']==118].head(10)


# In[23]:


plt.figure()
sns.distplot(profile['age'], bins=50, kde=False);


# **Hence, from the above analysis it's clear that people with age 118 are fake entries.**

# In[24]:


# check the gender

profile['gender'].value_counts()


# ## Transcript Dataset¶
# 

# In[25]:


transcript.head()


# - Transcript dataset contain -
# <ol>
#     <li> Person </li>
#     <li> Event </li>
#     <li> Value </li>
#     <li> Time </li>
# 
# 
# </ol>

# In[26]:


# Number of row in the dataset 

print("Number of row in the dataset are : ", transcript.shape[0])


# In[27]:


# Number of column in the dataset 

print("Number of column in the dataset are : ", transcript.shape[1])


# In[28]:


# data info

transcript.info()


# In[29]:


# check weather there is a null value or not

transcript.isnull().sum()


# **There are no null values.**

# In[30]:


# describe the dataset

transcript.describe()


# In[31]:


# datatype of dataframe

transcript.dtypes


# In[32]:


# check the type of events

transcript['event'].value_counts()


# # Data Preprocessing

# ## Portfolio Dataset

# In[34]:


portfolio_df = portfolio.copy()


# In[35]:


portfolio_df.head()


# In[36]:


# rename column id to offer_id

portfolio_df.rename(columns={'id':'offer_id'},inplace=True)


# In[37]:


# info of dataset

portfolio_df.info()


# In[38]:


# describe dataset

portfolio_df.describe()


# In[39]:


# change duration column in hours

portfolio_df['duration'] = portfolio_df['duration']*24


# In[40]:


# chnage the column name duration to duration_hours

portfolio_df.rename(columns={'duration':'duration_hours'}, inplace=True)


# In[41]:


scaler = MinMaxScaler() 

numerical = ['difficulty','reward']

portfolio_df[numerical] = scaler.fit_transform(portfolio_df[numerical])


# In[42]:


portfolio_df.head()


# In[43]:


# dataset info

portfolio_df.info()


# **Split the channel into diffrent column**

# Create dummy variables from 'channels'  column using one-hot encoding

# In[44]:


#  channel

portfolio_df['channels']


# In[45]:


# split channel into diffrent column

portfolio_df['channel_web'] =  portfolio_df['channels'].apply(lambda x: 1 if 'web' in x else 0)

portfolio_df['channel_email'] = portfolio_df['channels'].apply(lambda x: 1 if 'email' in x else 0)


portfolio_df['channel_social'] = portfolio_df['channels'].apply(lambda x: 1 if 'social' in x else 0)

portfolio_df['channel_mobile'] = portfolio_df['channels'].apply(lambda x: 1 if 'mobile' in x else 0)


# In[46]:


# check the column

portfolio_df[['channels','channel_email','channel_mobile','channel_web','channel_social']]


# In[47]:


# drop the column channel

portfolio_df.drop('channels', axis=1, inplace=True)


# In[48]:


portfolio_df.head()


# In[49]:


# dataset info

portfolio_df.info()


# In[50]:



# change the column name offer_id to id

offer_ids = portfolio_df['offer_id'].astype('category').cat.categories.tolist()

chnage_map_offer_ids = {'offer_id' : {k: v for k,v in zip(offer_ids,list(range(1,len(offer_ids)+1)))}}


# In[51]:


# check the new offer id


chnage_map_offer_ids


# In[52]:


# replacing the categorical values in the 'offer_id' column by numberical values


portfolio_df.replace(chnage_map_offer_ids, inplace=True)


# ### Preprocessing offer_type Feature

# In[53]:


# chnage  the 'offer_type' by integers 

offer_types = portfolio_df['offer_type'].astype('category').cat.categories.tolist()

chnage_map_offer_types = {'offer_type' : {k: v for k,v in zip(offer_types,list(range(1,len(offer_types)+1)))}}


# In[54]:


# check the new offer types

print(chnage_map_offer_types)


# In[55]:


# replacing the categorical values in the 'offer_type' column by numberical values



portfolio_df.replace(chnage_map_offer_types, inplace=True)


# In[56]:


# info

portfolio_df.info()


# ## Profile Dataset

# In[57]:


# creating a copy from the dataset to be cleaned


profile_df = profile.copy()


# In[58]:



profile_df.head()


# In[59]:


# Number of row in the dataset 

print("Number of row in the dataset are : ", profile_df.shape[0])


# In[60]:


# Number of column in the dataset 

print("Number of column in the dataset are : ", profile_df.shape[1])


# In[61]:


# data info

profile_df.info()


# In[62]:


# describe the dataset

profile_df.describe()


# In[63]:


# chnage the column name id to customer_id 


profile_df.rename(columns={'id':'customer_id'},inplace=True)


# In[64]:


# checking the existing columns' order


profile_df.info()


# In[65]:


# Re-arranging the columns


profile_df = profile_df.reindex(columns=['customer_id', 'age', 'became_member_on', 'gender', 'income'])


# In[66]:


# check the column index


profile_df.columns


# ### Preprocessing id Feature

# In[67]:


# change the customer_id from string to numerical value


customer_ids = profile_df['customer_id'].astype('category').cat.categories.tolist()

change_map_customer_ids = {'customer_id' : {k: v for k,v in zip(customer_ids,list(range(1,len(customer_ids)+1)))}}


# In[68]:


# replacing the  categorical labels in 'customer_id' column with numerical labels


profile_df.replace(change_map_customer_ids, inplace=True)


# ### Preprocessing  age Feature

# In[69]:


# change age value 118 by NaN


profile_df['age'] = profile_df['age'].apply(lambda x: np.nan if x == 118 else x)


# In[70]:




profile_df[profile_df['age'] == 118] 


# In[71]:


# drop the null value

profile_df.dropna(inplace=True)


# In[72]:


# check weather profile_df dataset has null value or not


profile_df.isna().sum()


# In[73]:


# changing the datatype of 'age' and 'income' columns to int


profile_df[['age', 'income']] = profile_df[['age', 'income']].astype(int)


# ### Age groups

# In[74]:


# create a new column that represent the different age group


profile_df['age_group'] = pd.cut(profile_df['age'], bins=[17, 22, 35, 60, 103],labels=['teenager', 'young-adult', 'adult', 'elderly'])


# In[75]:


profile_df['age_group']


# In[76]:


# replacing the 'age_group' categorical labels by numerical labels


age_groups = profile_df['age_group'].astype('category').cat.categories.tolist()


change_map_age_groups = {'age_group' : {k: v for k,v in zip(age_groups,list(range(1,len(age_groups)+1)))}}


# In[77]:


change_map_age_groups


# In[78]:


# change the categorical value to numerical value

profile_df.replace(change_map_age_groups, inplace=True)


# In[79]:




profile_df['age_group']


# **Preprocessing 'income' Feature**

# In[80]:


# create a new column for the icome range


profile_df['income_range'] = pd.cut(profile_df['income'], bins=[29999, 60000, 90000, 120001],labels=['average', 'above-average', 'high'])


# In[81]:


# replacing the 'income_range' categorical labels by numerical labels


income_ranges = profile_df['income_range'].astype('category').cat.categories.tolist()


change_map_income_ranges = {'income_range' : {k: v for k,v in zip(income_ranges,list(range(1,len(income_ranges)+1)))}}


# In[82]:


# checking the categorical labels and its corresponding numerical labels for 'income_range' column


change_map_income_ranges


# In[83]:


# replacing categorical labels in 'income_range' column with numerical labels


profile_df.replace(change_map_income_ranges, inplace=True)


# ### Gender feature

# In[84]:


# replacing the 'gender' categorical labels with coressponding numerical label


genders = profile_df['gender'].astype('category').cat.categories.tolist()

change_map_gender = {'gender' : {k: v for k,v in zip(genders,list(range(1,len(genders)+1)))}}


profile_df.replace(change_map_gender, inplace=True)


# In[85]:


# checking the numerical label and its corresponding categorical label


change_map_gender


# ### membership_days Feature 

# In[86]:


# chnage the datatype from int to date format

profile_df['became_member_on'] = pd.to_datetime(profile_df['became_member_on'], format = '%Y%m%d')


# In[87]:


# add new column start year


profile_df['membership_year'] = profile_df['became_member_on'].dt.year


# In[88]:


# adding a new column 'membership_days' ,that will present the number of days since the customer become a member


profile_df['membership_days'] = datetime.datetime.today().date() - profile_df['became_member_on'].dt.date


# In[89]:


profile_df['membership_days'] = profile_df['membership_days'].dt.days


# In[90]:


# creating a new column 'member_type' representing the type of the member: new, regular or loyal depending on the number of his 'membership_days'



profile_df['member_type'] = pd.cut(profile_df['membership_days'], bins=[390, 1000, 1600, 2500],labels=['new', 'regular', 'loyal'])


# In[91]:


# replacing the 'member_type' categorical labels by numerical labels


member_types = profile_df['member_type'].astype('category').cat.categories.tolist()


change_map_member_type = {'member_type' : {k: v for k,v in zip(member_types,list(range(1,len(member_types)+1)))}}


# In[92]:



change_map_member_type


# In[93]:


# replacing categorical labels in 'member_type' column with numerical labels


profile_df.replace(change_map_member_type, inplace=True)


# In[94]:


# drop the column

profile_df.drop(columns = ['age','income','became_member_on', 'membership_days'], axis=1, inplace=True)


# In[95]:




profile_df.head()


# In[96]:


# dataset info


profile_df.info()


# ## Transcript Dataset

# In[97]:


# create a copy from the dataset to be cleaned


transcript_df = transcript.copy()


# In[98]:


transcript_df.head()


# In[99]:


# Number of row in the dataset 


print("Number of row in the dataset are : ", transcript_df.shape[0])


# In[100]:


# Number of column in the dataset 


print("Number of column in the dataset are : ", transcript_df.shape[1])


# In[101]:


# describe the dataset


transcript_df.describe()


# In[102]:


# data info

transcript_df.info()


# In[103]:


# rename the column 


transcript_df.rename(columns={'person':'customer_id','time':'time_h' },inplace=True)


# In[104]:


# change the categorical value to numerical value


transcript_df.replace(change_map_customer_ids, inplace=True)


# In[105]:




transcript_df['customer_id']


# ### Preprocessing value Feature

# In[106]:




key = []
for index, row in transcript_df.iterrows():
    for i in row['value']:
        if i in key:
            continue
        else:
            key.append(i)


# In[107]:


key


# In[108]:


# create column and specify datatype

transcript_df['offer_id'] = '' 


# In[109]:


transcript_df['amount'] = 0  


# In[110]:


transcript_df['reward'] = 0  


# In[111]:



# update value by itrating the column


for index, row in transcript_df.iterrows():
    
    for i in row['value']:
        if i == 'offer_id' or i == 'offer id': 
            transcript_df.at[index, 'offer_id'] = row['value'][i]
        if i == 'amount':
            transcript_df.at[index, 'amount'] = row['value'][i]
        if i == 'reward':
            transcript_df.at[index, 'reward'] = row['value'][i]


# In[112]:


# filling all the NaNs in the 'offer_id' 


transcript_df['offer_id'] = transcript_df['offer_id'].apply(lambda x: 'N/A' if x == '' else x)


# In[113]:


# drop the value

transcript_df.drop('value', axis=1, inplace=True)


# ### Preprocessing event Feature

# In[114]:


# excluding all events of 'transaction' from our clean_transcript dataset


transcript_df = transcript_df[transcript_df['event'] != 'transaction']


# In[115]:


# excluding all events of 'offer received' 


transcript_df = transcript_df[transcript_df['event'] != 'offer received']


# In[116]:


# chnage the categorical value to numerical value


events = transcript_df['event'].astype('category').cat.categories.tolist()


chnage_map_events = {'event' : {k: v for k,v in zip(events,list(range(1,len(events)+1)))}}


# In[117]:


# checking the numerical label and its corresponding categorical label


chnage_map_events


# In[118]:


# replace categorical labels in 'event' column with numerical labels


transcript_df.replace(chnage_map_events, inplace=True)


# ### Preprocessing offer_id Feature

# In[119]:


# chnage the categeorical value to numerical value


transcript_df.replace(chnage_map_offer_ids, inplace=True)


# In[120]:




transcript_df.head()


# In[121]:


transcript_df.describe()


# # Combine the dataset

# In[122]:


# merge all the dataset


combine_df =transcript_df.merge(portfolio_df,how='left',on='offer_id')

combine_df = combine_df.merge(profile_df,how ='left', on = 'customer_id')


# In[123]:



combine_df.head()


# In[124]:


# info

combine_df.info()


# In[125]:


# Number of row in the dataset 

print("Number of row in the dataset are : ", combine_df.shape[0])


# In[126]:


# Number of column in the dataset 

print("Number of column in the dataset are : ", combine_df.shape[1])


# In[127]:


# desccribe

combine_df.describe()


# In[128]:


# drop the null value


combine_df = combine_df.dropna(how='any',axis=0) 


# In[129]:


# info

combine_df.info()


# 
# #  Data Analysis

# **The question that I'm intrested to answer are :**
# 
# **1-** Which Age group has heighest number of Customers?
# 
# **2-** How much offer are viewed and completed?
# 
# **3-** Which are the most popular offer.?
# 
# **4-** Most of the customer belongs to which income range?
# 
# **5-** Which type of promotions(offers) each gender likes?
# 
# 

# ## Which Age group has heighest number of Customers?

# - Mapping of Numerical values for  age_group -
#     - 1 - Teenager
#     - 2 - Young Adult
#     - 3 - Adult
#     - 4 - Elderly

# In[177]:


combine_df['age_group'] = combine_df['age_group'].map({1: 'teenager', 2: 'young-adult', 3:'adult', 4:'elderly'})


# In[178]:


combine_df.age_group.value_counts()


# In[179]:


plt.figure(figsize=(10,5))

combine_df.age_group.value_counts().reindex(['teenager', 'young-adult', 'adult', 'elderly']).plot(kind='bar', rot=0, figsize=(10,6), color='tab:red');
plt.ylabel('Number of People');
plt.grid();


# In[180]:


labels=['adult','elderly','young-adult','teenager']
values=[38845, 31302, 8656, 2851]

# adult          38845
# elderly        31302
# young-adult     8656
# teenager        2851

import matplotlib.pyplot as plt
explode=(0.10,0.10,0.10,0)

fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.show()


# **From the above graph its clear that most of the customer belong to adult age group.**
# 
#  - Number of Customer belong to Adult age group are **38845 i.e. 47.6%.**
#      
#  - Number of Customer belong to Elderly age group are **31302 i.e. 38.3%.**
#       
# - Number of Customer belong to Young Adult age group are **8656 i.e. 10.6%.**
# 
# - Number of Customer belong to Teenager age group are **2851 i.e. 3.5%.**

# ## How much offer are viewed and completed?

# - Mapping of Numerical values for  event -
#     - 1 - Offer Completed
#     - 2 - Offer Viewed
# 

# In[181]:


combine_df['event'] = combine_df['event'].map({1: 'Completed', 2: 'Viewed'})


# In[182]:


combine_df.event.value_counts()


# In[203]:


plt.figure(figsize=(10,5))

combine_df.event.value_counts().reindex(['Viewed', 'Completed']).plot(kind='bar', rot=0, figsize=(10,6), color='tab:green');
plt.ylabel('Number of People');
plt.grid();


# 
# 
# 
# **From the above graph it's clear that most the customer only viewed offer but only few completed the offer.**
#    - The number of customer Viewed offers are **49455.**
#    - The number of customer Completed offers are **32186.**
# 
# 

# ## Which are the most popular offer.?
# 

# - Mapping of Numerical values for  offer_type - 
#     - 1 - Bogo
#     - 2 - Discount
#     - 3 - Informational
# 

# In[184]:


combine_df['offer_type'] = combine_df['offer_type'].map({1: 'BOGO', 2: 'Discount', 3: 'Informational'})


# In[138]:


combine_df.offer_type.value_counts()


# In[204]:


plt.figure(figsize=(10,5))

combine_df.offer_type.value_counts().reindex(['BOGO', 'Discount', 'Informational']).plot(kind='bar', rot=0, figsize=(10,6), color='tab:orange');
plt.ylabel('Number of People');
plt.grid();


# In[140]:


labels=['Discount','Bogo','International']
values=[35357,  37006, 9291]
import matplotlib.pyplot as plt
explode=(0.20,0.20,0)

fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.show()


# **From the above graph it's clear that Bogo offer are most popular among the customers.**
# 
# 
#    - BOGO offer with **37002 i.e. 45.3%**
#    - Discount offer with **35349 i.e. 43.3%**
#    - Informational offer with **9290 i.e. 11.4%**

# ## Most of the customer belongs to which income range?

#         
# - Mapping of Numerical values for  income_range -
#     - 1 - Average 
#     - 2 - Above-average 
#     - 3 - High (more than 90k)
# 
# 

# In[186]:


combine_df['income_range'] = combine_df['income_range'].map({1: 'Average', 2: 'Above-Average', 3:'High'})


# In[187]:


combine_df.income_range.value_counts()


# In[207]:


plt.figure(figsize=(10,5))

combine_df.income_range.value_counts().reindex(['Above-Average', 'Average', 'High']).plot(kind='bar', rot=0, figsize=(10,6), color='tab:purple');
plt.ylabel('Number of People');
plt.grid();


# In[189]:


labels=['Above-Average ','Average','High']
values=[34844,  33627, 13183]
import matplotlib.pyplot as plt
explode=(0.20,0.20,0)



fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.show()


# **From the above graph it is clear that most of the customer belong to above average income range i.e 42.7%.**
# 
# - Number of customer belongs to Average income range are  **34844 i.e. 41.2%.**
# - Number of customer belongs to Above-Average income range are  **33623 i.e. 42.7%.**
# - Number of customer belongs to High income range are   **13174 i.e. 16.1%.**

# ## Which type of offers each gender like ? 

# 
# - Mapping of Numerical values for  gender -
#     - 1 - Female
#     - 2 - Male
#     - 3 - Other
# 

# In[190]:


plt.figure(figsize=(9, 5))
g = sns.countplot(x='gender', hue="offer_type", data= combine_df[combine_df["gender"] != 3])
plt.title('Most Popular Offers to Each Gender')
plt.ylabel('Total')
plt.xlabel('Gender')
xlabels = ['Female', 'Male']
g.set_xticklabels(xlabels)
plt.legend(title='Offer Type')
plt.show();


# 
# **From the above graph we can see that Bogo offer liked by both male and female. Information offfer are least liked by both male and female.**

# In[156]:


# Replacing the categorical values of the features by its corresponding numerical values, as before
events_new = combine_df['event'].astype('category').cat.categories.tolist()
change_map_events_new = {'event' : {k: v for k,v in zip(events_new,list(range(1,len(events_new)+1)))}}

income_ranges_new = combine_df['income_range'].astype('category').cat.categories.tolist()
chnage_map_income_new = {'income_range' : {k: v for k,v in zip(income_ranges_new,list(range(1,len(income_ranges_new)+1)))}}

offer_types_new = combine_df['offer_type'].astype('category').cat.categories.tolist()
chnage_map_offer_types_new = {'offer_type' : {k: v for k,v in zip(offer_types_new,list(range(1,len(offer_types_new)+1)))}}

combine_df.replace(change_map_events_new, inplace=True)
combine_df.replace(chnage_map_offer_types_new, inplace=True)
combine_df.replace(chnage_map_income_new, inplace=True)
combine_df.replace(change_map_age_groups, inplace=True)


# In[157]:


# confirming changes
combine_df.head()


# # Data Modeling

# In[158]:


combine_df.columns


# In[159]:


# Rename 'reward_x' column to 'reward'
combine_df.rename(columns ={'reward_x':'reward'}, inplace = True)


# In[160]:


# Split the data into features and target label

# Features are -


# time_h, offer_id, amount, reward_x ( Will be renamed to 'reward'), difficulty, duration_h, 
# offer_type, gender, age_group, income_range, member_type


# Target are - 

# event

X = combine_df[['time_h','offer_id','amount','reward','difficulty','duration_hours','offer_type','gender','age_group','income_range', 'member_type']]
Y = combine_df['event']


# In[161]:


X.head()


# In[162]:


Y.head()


# In[163]:



scaler = MinMaxScaler()

features = ['time_h', 'amount', 'reward', 'duration_hours']

X_scaled = X.copy()

X_scaled[features] = scaler.fit_transform(X_scaled[features])

X_scaled.head()


# ### Metrices
#  
# - In order to evaluate our model performance we are using the accuracy.
# 
#     Here are the following reason for choosing this metrices -
# 
# - We are using the simple classification problem 
#     - 1) Offer viewed
#     - 2) Offer completed
# - To know how our model is performing we will compare our model with number of correct prediction to total number of prediction (accuracy).
# 

# In[164]:


# creating training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)


# As we mentioned above -
# 
# - In order to evaluate our model performance we are using the accuracy.
# - We are using the simple classification problem 
#     - 1) Offer viewed
#     - 2) Offer completed
# - To know how our model is performing we will compare our model with number of correct prediction to total number of prediction (accuracy).

# In[165]:


# defining a function to calculate the accuracy for the models we will try below 
def predict_score(model):
    pred = model.predict(X_test)
    
    # Calculate the absolute errors
    errors = abs(pred - y_test)
    
    # Calculate mean absolute percentage error
    mean_APE = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mean_APE)
    
    return round(accuracy, 4)


# ### Support Vector Machine.

# In[191]:


from sklearn import svm
svm = svm.SVC(gamma = 'auto')

svm.fit(X_train, y_train)
print(f'Accuracy of SVM classifier on training set are - {round(svm.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy are - {predict_score(svm)}%')


# ### Decision Tree.

# In[212]:


dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)
print(f'Accuracy of Decision Tree classifier on training set are - {round(dtree.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy are - {predict_score(dtree)}%')


# ### Naive Bayes.

# In[211]:


gus = GaussianNB() 
gus.fit(X_train, y_train) 
print(f'Accuracy of Naive classifier on training set are - {round(gus.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy are - {predict_score(gus)}%')


# ### K-Nearest Neighbors.

# In[194]:


knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
print(f'Accuracy of K-NN classifier on training set are -  {round(knn.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy are -  {predict_score(knn)}%')


# ### Random Forest.

# In[209]:


rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(X_train, y_train)
print(f'Accuracy of Random Forest classifier on training set are -  {round(rf.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy are - {predict_score(rf)}%')


# ### Logistic Regression.

# In[210]:


log = LogisticRegression()

log.fit(X_train, y_train)
print(f'Accuracy of Logistic regression classifier on training set are -  {round(log.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy are - {predict_score(log)}%')


# ## Model Evaluation

# In[197]:


# creating the variables that will be used to fill the results table


model = [svm, dt, gnb, knn, rf, logreg]

models_re = [type(n).__name__ for n in model]

training_accuracy = [x.score(X_train, y_train)*100 for x in model]

predection_accuracy = [predict_score(y) for y in model]


# In[201]:



results = [training_accuracy, predection_accuracy]

df_results = pd.DataFrame(results, columns = models_re, index=['Training Accuracy', 'Predicting Accuracy']) 


# In[202]:


# show the results dataframe 
df_results


# **From the above table we can see that 4 out 6 model give 100% accuracy. This will leads to over fitting.**
# - To avoid the over fitting we will choose the model that will give the lowest accuracy score.
# - The model that give lowest accuracy are **KNeighborsClassifier** It will give the accuracy of 99.9395% In training and 99.9314% in prediction.
# - Although its very high accuracy. But we will go with the lowest one.

# ## Model Refinment

# I personally belive that nothing is perfect so everything can improved further. But now in this case i think **KNeighborsClassifier** model is satisfying. Becasuse it already gives us a very high score and and further if we tried to improve this then this will leads to overfitting.

# # Conclusion

# **StarBuck capstone project - In this project I cleaned the dataset and made some improvement to analyize the dataset and try to answer diffrent questions and make a model that predict that weather customer completed the offer or just only view the offer.**
# 

# **The quick analysis of the dataset are -** 
# 
# 
# **Q1-** **Which Age group has heighest number of Customers?**
# 
# - Number of Customer belong to Adult age group are **38845 i.e. 47.6%.**
#      
# - Number of Customer belong to Elderly age group are **31302 i.e. 38.3%.**
#       
# - Number of Customer belong to Young Adult age group are **8656 i.e. 10.6%.**
# 
# - Number of Customer belong to Teenager age group are **2851 i.e. 3.5%.**
# 
# **Q2- How much offer are viewed and completed?**
# 
# - The number of customer Viewed offers are **49455.**
# - The number of customer Completed offers are **32186.**
# 
# 
# 
# **Q3- Which are the most popular offer.?**
# 
# - BOGO offer with **37002 i.e. 45.3%**
# - Discount offer with **35349 i.e. 43.3%**
# - Informational offer with **9290 i.e. 11.4%**
# 
# 
# **Q4- Most of the customer belongs to which income range?**
# 
# - Number of customer belongs to Average income range are  **34844 i.e. 41.2%.**
# - Number of customer belongs to Above-Average income range are  **33623 i.e. 42.7%.**
# - Number of customer belongs to High income range are   **13174 i.e. 16.1%.**
# 
# **Q5- Which type of promotions(offers) each gender likes?**
# 
# - From the above graph we can see that Bogo offer liked by both male and female. Information offfer are least liked by both male and female.
# 

# In[ ]:





# In[ ]:




