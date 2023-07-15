#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os # paths to file
import numpy as np # math
import pandas as pd # data processes
import warnings # warnings
import scipy as sp # pivot engineering

from sklearn.metrics.pairwise import cosine_similarity # model

pd.options.display.max_columns # default theme and settings


# In[2]:


# setting warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")


# In[3]:


# paths and get csvs
rating_path = "kaggle/kaggle/input/anime-recommendations-database/rating.csv"
anime_path = "kaggle/kaggle/input/anime-recommendations-database/anime.csv"
#rating gives user data, anime gives general info
rating_df = pd.read_csv(rating_path)
anime_df = pd.read_csv(anime_path)

print(rating_df.head())
print(anime_df.head())


# In[4]:


# printing data info
print(f"anime set (row, col): {anime_df.shape}\n\nrating set (row, col): {rating_df.shape}")
print("Anime:\n")
print(anime_df.info())
print("\n","*"*50,"\nRating:\n")
print(rating_df.info())


# In[5]:


# finding missing values to clear
print("Anime missing values (%):\n")
print(round(anime_df.isnull().sum().sort_values(ascending=False)/len(anime_df.index),4)*100) 
print("\n","*"*50,"\n\nRating missing values (%):\n")
print(round(rating_df.isnull().sum().sort_values(ascending=False)/len(rating_df.index),4)*100)


# In[6]:


# most common type and genre
print(anime_df['type'].mode())
print(anime_df['genre'].mode()) # haha mostly hentai


# In[7]:


# handling missing data
# deleting anime with 0 rating
anime_df=anime_df[~np.isnan(anime_df["rating"])]

# filling mode value for genre and type (guess we're filling it with hentai and TV huh)
anime_df['genre'] = anime_df['genre'].fillna(anime_df['genre'].dropna().mode().values[0])
anime_df['type'] = anime_df['type'].fillna(anime_df['type'].dropna().mode().values[0])

#checking if all null values are filled
print(anime_df.isnull().sum())

#replace values that have -1 rating with NaN
f = (lambda x: np.nan if x==-1 else x)
rating_df['rating'] = rating_df['rating'].apply(f)
print(rating_df.head(20))


# In[8]:


# engineering dataframes
anime_df = anime_df[anime_df['type'] == 'TV']

# merges and coordinates, suffixes adds the name of things on left and on right to rating
rated_anime = rating_df.merge(anime_df, left_on = 'anime_id', right_on = 'anime_id', suffixes = ['_user', ''])
#print(rated_anime)

# isolate these columns
rated_anime = rated_anime[['user_id', 'name', 'rating']]

# limiting 
rated_anime_10000 = rated_anime[rated_anime.user_id <= 7500]
rated_anime_10000.head()


# In[9]:


# pivots by user_id x name and makes the values of table ratings (easier to read)
pivot = rated_anime_10000.pivot_table(index=['user_id'], columns=['name'], values='rating')
pivot.head()


# In[10]:


# value normalization through min-max scaling
f = lambda x: (x-np.mean(x))/(np.max(x)-np.min(x))
pivot_n = pivot.apply(f, axis=1) # axis changes how lambdas functions (1 is across columns)

pivot_n.fillna(0, inplace=True) # inplace true replaces data frame
pivot_n = pivot_n.transpose() # reflects the data

#drops column with values of 0
pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]

#converts to sparse matrix format (just a matrix format where lots of 0s are there to process)
piv_sparse = sp.sparse.csr_matrix(pivot_n.values)
pivot_n.head()


# In[11]:


# model based on anime similarity
anime_similarity = cosine_similarity(piv_sparse)

#convert to data frame of anime similarities
ani_sim_df = pd.DataFrame(anime_similarity, index = pivot_n.index, columns = pivot_n.index)


# In[12]:


# method for anime_recommendation
def anime_recommendation(ani_name):
    if (anime_df['name'].eq(ani_name)).any():
        number = 1
        ans = ""
        print('Recommended because you watched {}:\n'.format(ani_name))
        ans = []
        ans.append('Recommended because you watched {}:\n'.format(ani_name))
        for anime in ani_sim_df.sort_values(by = ani_name, ascending = False).index[1:6]:
            print(f'#{number}: {anime}, {round(ani_sim_df[anime][ani_name]*100,2)}% match')
            ans.append(f'#{number}: {anime}, {round(ani_sim_df[anime][ani_name]*100,2)}% match\n')
            number +=1  
        return ans
    else:
        return ["Sorry!, couldn't find any anime named that!"]







