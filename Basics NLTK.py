#!/usr/bin/env python
# coding: utf-8

# In[23]:


import nltk
from nltk.corpus import brown
from nltk.tokenize import WhitespaceTokenizer
nltk.download('punkt')
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WhitespaceTokenizer
nltk.download('stopwords')

with open('C:/Users/Chandan/Documents/CPTS 570/NLTK/code/datasets/biography.txt', 'r') as f:
    data = f.read()

print(data)


# In[24]:


from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

print(stop_words)


# In[27]:


from nltk.probability import FreqDist
import matplotlib.pyplot as plt
word_tokens = word_tokenize(data)
freq_dist = FreqDist(word_tokens)
fig, ax = plt.subplots(figsize=(12, 8))

freq_dist.plot(20, cumulative=False)

plt.show()


# In[29]:


filtered_words = []

for w in word_tokens:
    if w not in stop_words:
        filtered_words.append(w)
        
print(filtered_words)

freq_dist1 = FreqDist(filtered_words)
fig, ax = plt.subplots(figsize=(12, 8))

freq_dist1.plot(20, cumulative=False)

plt.show()


# In[ ]:




