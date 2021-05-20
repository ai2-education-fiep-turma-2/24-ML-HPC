#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/home/silvio/dataset/stackoverflow'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[4]:


import datetime

B=datetime.datetime.now()
df_train=pd.read_csv('/home/silvio/dataset/stackoverflow/train.csv')
#df_train=pd.read_csv('/home/silvio/dataset/stackoverflow/train-sample.csv')
df_train.head()
E=datetime.datetime.now()
print(E-B)


# In[ ]:


df_train.tail(13)


# In[ ]:


pd.set_option('display.max_colwidth', None)


# In[ ]:


df_train['Title'].head(2)


# In[ ]:


df_train['BodyMarkdown'].head(2)


# In[ ]:





# In[ ]:


df_train['OpenStatus']


# In[ ]:


df_train['OpenStatus'].unique()


# In[ ]:


df_train.info()


# In[ ]:


df_train.shape


# In[5]:


B=datetime.datetime.now()

from sklearn.model_selection import train_test_split
train_df,test_df=train_test_split(df_train,test_size=0.2)
print('Training data shape: {}'.format(train_df.shape))
print('Testing data shape: {}'.format(test_df.shape))

E=datetime.datetime.now()
print(E-B)


# # Exploratory Data Analysis

# ## PostID

# This feature is the id of the query posted by the user. This feauture does not have any importance in making the prediction but it will be used later for submission we will save it.

# In[6]:


train_post_id=train_df.PostId
test_post_id=test_df.PostId


# ## PostCreationDate
# 
# The post creation date is the date and time on which the query was posted by the user let us see if we could make something out of this.

# As we have observed that it doesn't matter on the time the question is posted on its label thus we can just remove this column both from our train and test.

# In[7]:


train_df.drop(['PostCreationDate'],axis=1,inplace=True)
test_df.drop(['PostCreationDate'],axis=1,inplace=True)


# In[8]:


train_df.head(2)


# ## OwnerUserId

# This is the user id who posted the query. Let us check it

# In[9]:


train_df['PostCount']=[1]*len(train_df['PostId'])


# In[10]:


pd.pivot_table(train_df,index='OwnerUserId',columns='OpenStatus',values='PostCount',aggfunc='sum')


# In[11]:


train_df.loc[train_df['OpenStatus']=='open','OwnerUserId'].value_counts()


# We see that it does not matter on the OwnerUserId whether his question is going to be open or not as there are many user whose only one question remain open 
# So, we would be dropping this column

# In[12]:


train_df.drop(['OwnerUserId'],axis=1,inplace=True)
test_df.drop(['OwnerUserId'],axis=1,inplace=True)


# ## Reputation at Post Created
# 
# This can be a important factor. Let's take a look at it

# In[13]:


df_train['ReputationAtPostCreation'].min()


# In[14]:


df_train['ReputationAtPostCreation'].max()


# Since the data range is such high let us scale it using MinMaxScaler

# In[15]:


minimum=df_train['ReputationAtPostCreation'].min()
maximum=df_train['ReputationAtPostCreation'].max()


# In[16]:


train_df['ReputationAtPostCreation']=(train_df['ReputationAtPostCreation']-minimum)/(maximum-minimum)


# In[17]:


test_df['ReputationAtPostCreation']=(test_df['ReputationAtPostCreation']-minimum)/(maximum-minimum)


# ## OwnerUndeletedAnswerCountAtPostTime 

# In[18]:


train_df.OwnerUndeletedAnswerCountAtPostTime.value_counts()


# This feature doesnot seem to be doing anything, we will be dropping it for now.

# In[19]:


train_df.drop(['OwnerUndeletedAnswerCountAtPostTime'],axis=1,inplace=True)
test_df.drop(['OwnerUndeletedAnswerCountAtPostTime'],axis=1,inplace=True)


# ## PostClosedDate

# Dropping this column as post closed date does not seem to be doing anything, so we would be dropping it 

# In[20]:


train_df.drop(['PostClosedDate'],axis=1,inplace=True)
test_df.drop(['PostClosedDate'],axis=1,inplace=True)


# ## Tag1

# In[21]:


train_df['Tag1'].isnull().sum()


# In[22]:


test_df['Tag1'].isnull().sum()


# We can combine all the tags column into one this would help us in comparing the words which are used as tags and the words used in Title and Body.

# In[23]:


train_df['Tag1']=train_df['Tag1'].replace(np.nan,' ')


# In[24]:


train_df['Tag2']=train_df['Tag2'].replace(np.nan,' ')
train_df['Tag3']=train_df['Tag3'].replace(np.nan,' ')
train_df['Tag4']=train_df['Tag4'].replace(np.nan,' ')
train_df['Tag5']=train_df['Tag5'].replace(np.nan,' ')


# In[25]:


test_df['Tag1']=test_df['Tag1'].replace(np.nan,' ')
test_df['Tag2']=test_df['Tag2'].replace(np.nan,' ')
test_df['Tag3']=test_df['Tag3'].replace(np.nan,' ')
test_df['Tag4']=test_df['Tag4'].replace(np.nan,' ')
test_df['Tag5']=test_df['Tag5'].replace(np.nan,' ')


# In[26]:


train_df['Tag1']


# In[27]:


train_df['Tags']=train_df['Tag1']+' '+train_df['Tag2']+' '+train_df['Tag3']+' '+train_df['Tag4']+' '+train_df['Tag5']


# In[28]:


test_df['Tags']=test_df['Tag1']+' '+test_df['Tag2']+' '+test_df['Tag3']+' '+test_df['Tag4']+' '+test_df['Tag5']


# In[29]:


train_df['Tags']=train_df['Tags'].str.lower()


# In[30]:


test_df['Tags']=test_df['Tags'].str.lower()


# In[31]:


test_df['Tags']


# In[32]:


train_df['Tags']=train_df['Tags'].apply(lambda x:x.lstrip())
train_df['Tags']=train_df['Tags'].apply(lambda x:x.rstrip())


# In[33]:


train_df['Tags']


# In[34]:


test_df['Tags']=test_df['Tags'].apply(lambda x:x.lstrip())
test_df['Tags']=test_df['Tags'].apply(lambda x:x.rstrip())


# In[35]:


train_df.head(1)


# In[36]:


# Dropping excess columns 
train_df.drop(['PostId','OwnerCreationDate','Tag1','Tag2','Tag3','Tag4','Tag5','PostCount'],axis=1,inplace=True)
test_df.drop(['PostId','OwnerCreationDate','Tag1','Tag2','Tag3','Tag4','Tag5'],axis=1,inplace=True)


# In[37]:


train_df.head()


# In[38]:


test_df.head()


# In[ ]:





# In[39]:


y_train=train_df['OpenStatus']
y_test=test_df['OpenStatus']


# In[40]:


train_df.drop(['ReputationAtPostCreation','OpenStatus'],axis=1,inplace=True)
test_df.drop(['ReputationAtPostCreation','OpenStatus'],axis=1,inplace=True)


# In[41]:


y_train=y_train.map({'not a real question':0,
  'not constructive':1,
  'off topic':2,
  'open':3,
  'too localized':4})


# In[42]:


y_test=y_test.map({'not a real question':0,
  'not constructive':1,
  'off topic':2,
  'open':3,
  'too localized':4})


# In[43]:


y_test.unique()


# In[44]:


train_df.head()


# In[45]:


test_df.head()


# In[46]:


train_df['Text']=train_df['Title']+' '+train_df['BodyMarkdown']+' '+train_df['Tags']
test_df['Text']=test_df['Title']+' '+test_df['BodyMarkdown']+' '+test_df['Tags']


# In[47]:


train_df.drop(['Title','BodyMarkdown','Tags'],axis=1,inplace=True)
test_df.drop(['Title','BodyMarkdown','Tags'],axis=1,inplace=True)


# In[48]:


train_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)


# In[49]:


train_df.drop(['index'],inplace=True,axis=1)


# In[50]:


test_df.drop(['index'],axis=1,inplace=True)


# In[51]:


train_df.head()


# # Building the tensorflow model
# 

# In[ ]:
'''

from tensorflow.keras.preprocessing.text import Tokenizer
max_words=10000
tokenizer=Tokenizer(max_words)
tokenizer.fit_on_texts(train_df['Text'])
sequence_train=tokenizer.texts_to_sequences(train_df['Text'])
sequence_test=tokenizer.texts_to_sequences(test_df['Text'])


# In[ ]:


word_2_vec=tokenizer.word_index
V=len(word_2_vec)
print('Dataset has {} number of independent tokens'.format(V))


# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
data_train=pad_sequences(sequence_train)
data_train.shape


# In[ ]:


T=data_train.shape[1]
data_test=pad_sequences(sequence_test,maxlen=T)
data_test.shape


# In[ ]:


from tensorflow.keras.layers import Input,Conv1D,MaxPooling1D,Dense,GlobalMaxPooling1D,Embedding
from tensorflow.keras.models import Model
D=20
i=Input((T,))
x=Embedding(V+1,D)(i)
x=Conv1D(32,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(64,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation='relu')(x)
x=GlobalMaxPooling1D()(x)
x=Dense(5,activation='softmax')(x)
model=Model(i,x)
model.summary()


# In[ ]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
cnn_senti=model.fit(data_train,y_train,validation_data=(data_test,y_test),batch_size=100,epochs=5)


# In[ ]:


model.predict(data_test)


# In[ ]:


y_pred=model.predict(data_test)


# In[ ]:


y_pred_final=np.argmax(y_pred,axis=1)
y_pred_final


# ## Scoring

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns


# In[ ]:


cm=confusion_matrix(y_test,y_pred_final)
ax=sns.heatmap(cm,cmap='Blues',annot=True,fmt=' ')
ax.set_title('Confusion Matrix')
ax.set_xlabel('Y Test')
ax.set_ylabel('Y Pred')


# In[ ]:


print(classification_report(y_test,y_pred_final))


# In[ ]:


df_submission=pd.DataFrame(test_post_id,columns=['PostId'])
df_submission.head()


# In[ ]:


test_post_id.reset_index(drop=True)


# In[ ]:


df_submission_1=pd.DataFrame(y_pred,columns=[0,1,2,3,4])


# In[ ]:


df_submission_1


# In[ ]:


df_submission_1['PostId']=list(test_post_id)


# In[ ]:


columns=['PostId',0,1,2,3,4]
df_submission_1=df_submission_1[columns]
df_submission_1


# In[ ]:


df_submission_1.columns


# In[ ]:


df_submission_1['Sum']=df_submission_1[0]+df_submission_1[1]+df_submission_1[2]+df_submission_1[3]+df_submission_1[4]


# In[ ]:


df_submission_1

'''
