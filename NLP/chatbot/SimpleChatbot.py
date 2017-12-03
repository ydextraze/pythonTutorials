
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import csv
import sys
import hashlib
import re
import string
import itertools
# use natural language toolkit

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

lemma = nltk.wordnet.WordNetLemmatizer()
sno = nltk.stem.SnowballStemmer('english')


# In[2]:

train = pd.read_csv(r'C:\Users\piush\Desktop\chatbot\sentences.csv')



def stematize(sentence):
    """
    pass  in  a sentence as a string, return just core text stemmed
    stop words are removed - could effect ability to detect if this is a question or answer
    - depends on import sno = nltk.stem.SnowballStemmer('english') and from nltk.corpus import stopwords
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(sentence)
    
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    stemmed = []        
    for w in filtered_sentence:
        stemmed.append(sno.stem(w))
  
    return stemmed    


# In[7]:

train['wordCount'] = train.apply(lambda row: len(row['SENTENCE'].split()), axis=1)


# In[12]:

train['stemmed']  =  train.apply(lambda row: stematize(row['SENTENCE']), axis=1)


# In[14]:

train['stemmedCount'] = train.apply(lambda row: len(row['stemmed']), axis=1)


# ###### Rearrange the columns

# In[26]:

train = train[['SENTENCE', 'wordCount', 'text_without_stopwords', 'stemmed',
       'stemmedCount', 'stemmedEndNN', 'pos' , 'CLASS']]


# In[27]:

# from sklearn.preprocessing import LabelEncoder
# class_le = LabelEncoder()
# train['text_without_stopwords'] = class_le.fit_transform(train['text_without_stopwords'].values)
# train['SENTENCE'] = class_le.fit_transform(train['SENTENCE'].values)
# train['stemmed'] = class_le.fit_transform(train['lemma'].values)
# train['get_triples'] = class_le.fit_transform(train['get_triples'].values)
# train['CLASS'] = class_le.fit_transform(train['CLASS'].values)


# #### Feature Selection

# In[28]:

train.columns


# In[29]:

train_subset = train[['wordCount', 'stemmed','stemmedCount', 'CLASS']]


# In[30]:

train_subset['stemmed']


# ##### One hot encoding the list (stemmed)

train_subset = train_subset.drop('stemmed', 1).join(train_subset.stemmed.str.join('|').str.get_dummies())


