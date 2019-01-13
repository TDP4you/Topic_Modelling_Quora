# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 23:55:48 2019

@author: tdpco
"""

# importing libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# importing dataset
dataset = pd.read_csv('quora_questions.csv')
print(dataset.head())

# Preprocessing
# Using Tfidf creating a vectorized document term matrix.
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(dataset['Question'])
# print(dtm)

# Non Negative Matrix Factorization
nmf_model = NMF(n_components=10, random_state=20)
nmf_model.fit(dtm)
# print(nmf_model)


# Printing top 15 most common word for each of 10 Topics
for index, topic in enumerate(nmf_model.components_):
    print(f"The top 15 common words for topic: {index+1}")
    print([tfidf.get_feature_names()[index] for index in topic.argsort()[-10:]])
    print('\n')

topic_result = nmf_model.transform(dtm)

# Assigning Question the topic number
dataset['Topic_No.'] = topic_result.argmax(axis=1) + 1

# Labelling these topics
topics_list = {1: 'Knowledge',
               2: 'Feeling',
               3: 'Suggestion',
               4: 'Online',
               5: 'Positivity',
               6: 'Investment',
               7: 'Language',
               8: 'U.S. Election',
               9: 'Demonetisation',
               10: 'Daily Life'}

# Mapping the topics to dataset
dataset['Topic'] = dataset['Topic_No.'].map(topics_list)
print(dataset.head(10))
