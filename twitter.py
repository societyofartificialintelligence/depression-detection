import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys

col_names = ["target", "id", "date", "flag", "user", "Tweet_text"]
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', names = col_names)
data.head()

data.drop(['id', 'date', 'flag', 'user'],axis = 1, inplace = True)
# check the unique values of target
data['target'].unique()

#Data Cleaning

# removing links
data['CleanText'] = data['CleanText'].str.replace(r"http\S+","")

data['CleanText'] = data['CleanText'].str.replace("[^a-zA-Z]"," ")

# remove stopwords
import re
import string

import nltk
from nltk.corpus import stopwords

print(stopwords.words('english'))

stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    clean_text=' '.join([word for word in text.split() if word not in stopwords])
    return clean_text


data['CleanText'] = data['CleanText'].apply(lambda text : remove_stopwords(text.lower()))
#Tokenization and normalization of data
data['CleanText'] = data['CleanText'].apply(lambda x: x.split())

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
data['CleanText'] = data['CleanText'].apply(lambda x:[stemmer.stem(i) for i in x])

data['CleanText'] = data['CleanText'].apply(lambda x: ' '.join([w for w in x]))

# count vectorization
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english') 
cv = count_vectorizer.fit_transform(data['CleanText'])
pickle.dump(cv,open('transform.pkl','wb'))

# splitting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(cv,data['target'] , test_size=.2,stratify=data['target'], random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

lr = LogisticRegression()
lr.fit(X_train,y_train)
prediction_lr = lr.predict(X_test)
print(accuracy_score(prediction_lr,y_test))
pickle.dump(cv,open('model.pkl','wb'))