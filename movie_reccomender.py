#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movie_ds = pd.read_csv('movies.csv')
credits_ds = pd.read_csv('credits.csv')


# In[4]:


movie_ds = movie_ds.merge(credits_ds, on='id')


# In[6]:


# genres, id, keywords, title, overview, cast, crew
movie_ds = movie_ds[['id', 'original_title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# In[9]:


movie_ds = movie_ds.dropna()


# In[12]:


genre = movie_ds.iloc[0].genres


# In[13]:


import ast
def convert(obj):
    L = []
    for i in ast.literal_eval(obj): 
        L.append(i['name'])
    return L


# In[15]:


movie_ds['genres'] = movie_ds['genres'].apply(convert)


# In[18]:


movie_ds['keywords'] = movie_ds['keywords'].apply(convert)


# In[21]:


movie_ds['cast'] = movie_ds['cast'].apply(convert)


# In[24]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[25]:


movie_ds['crew'] = movie_ds['crew'].apply(fetch_director)


# In[28]:


movie_ds['overview'] = movie_ds['overview'].apply(lambda x: x.split())


# In[30]:


movie_ds['genres'] = movie_ds['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movie_ds['keywords'] = movie_ds['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movie_ds['cast'] = movie_ds['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movie_ds['crew'] = movie_ds['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


# In[32]:


movie_ds['tags'] = movie_ds['overview'] + movie_ds['genres'] + movie_ds['keywords'] + movie_ds['cast'] + movie_ds['crew']


# In[34]:


new_ds = movie_ds[['id', 'original_title', 'tags']]


# In[36]:


new_ds['tags'] = new_ds['tags'].apply(lambda x: " ".join(x))


# In[38]:


new_ds['tags'] = new_ds['tags'].apply(lambda x: x.lower())


# In[41]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[42]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[43]:


new_ds['tags'] = new_ds['tags'].apply(stem)


# ## Vectorization

# In[44]:


#using bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[45]:


vector = cv.fit_transform(new_ds['tags']).toarray()


# In[47]:


cv.get_feature_names_out()


# In[48]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)


# In[49]:


def reccomend(movie):
    movie_index = new_ds[new_ds['original_title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_ds.iloc[i[0]].original_title)


# In[51]:


import pickle
pickle.dump(new_ds.to_dict(), open('movies_dict.pkl', 'wb'))


# In[52]:


pickle.dump(similarity, open('similarity.pkl', 'wb'))


# In[ ]:




