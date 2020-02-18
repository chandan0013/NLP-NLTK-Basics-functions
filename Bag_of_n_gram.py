#!/usr/bin/env python
# coding: utf-8

# In[5]:


#### SK learn is easy to understand first
import sklearn
nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer


# In[13]:


with open('C:/Users/Chandan/Documents/CPTS 570/NLTK/code/datasets/biography.txt', 'r') as f:
    data = f.read()
lines = data.split('\n')
print(data)
count_vectorizer = CountVectorizer()
n_gram_vectorizer = CountVectorizer(ngram_range=(2, 2))  ###can change range
transformed_vector = n_gram_vectorizer.fit_transform(lines)


# In[15]:


feature = n_gram_vectorizer.vocabulary_

transformed_vector.toarray()[:1]


# In[16]:


n_gram_vectorizer.vocabulary_


# In[17]:


feature.items()


# In[19]:


##### now using NLTK

from nltk import bigrams
from nltk import trigrams

from nltk.tokenize import word_tokenize


word_tokens = word_tokenize(" ".join(lines))

word_tokens


# In[20]:


nltk_bigrams = bigrams(word_tokens)

list(nltk_bigrams)


# In[21]:


nltk_trigrams = trigrams(word_tokens)

list(nltk_trigrams)


# In[22]:


from nltk.collocations import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
word_tokens = word_tokenize(data)


# In[23]:


bigram_measures = BigramAssocMeasures()

finder = BigramCollocationFinder.from_words(word_tokens)

finder.apply_freq_filter(3)

matches = finder.nbest(bigram_measures.raw_freq, 15)

matches


# In[ ]:




