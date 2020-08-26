import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

df=pd.read_csv("export(2).csv")
df.columns

df["label"] = np.where(df['count'] > 0,1,0)
    
df['low']=df['Test'].values
df['low']=df['low'].apply(lambda x:x.lower())
df['low'].head()

tokenizer=RegexpTokenizer(r'\w+')
df['token']=df['low'].apply(lambda row:tokenizer.tokenize(row))
df['token'].head()

stop_words=set(stopwords.words('english'))
df['stop_filter_text']=df['token'].apply(lambda row: [x for x in row if not x in stop_words])
df['stop_filter_text'].head()

import re
def remove_no(list): 
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list] 
    return list

df['stop_filter_text1'] = df['stop_filter_text'].apply(remove_no)

def remove_space(test_list):
    while("" in test_list) : test_list.remove("")
    return test_list

df['stop_filter_text1'].head()=df['stop_filter_text1'].apply(remove_space)

lemmatizer=WordNetLemmatizer()
df['lemmat_text']=df['stop_filter_text1'].apply(lambda x:[lemmatizer.lemmatize(a) for a in x])
df['lemmat_text'].head()

data=df.copy()
sentences=data['lemmat_text']
y=data['label']
sentences=sentences.astype(str)
train_x,test_x,train_y,test_y=train_test_split(sentences,y,test_size=0.3, random_state=500)

vectorizer=TfidfVectorizer(ngram_range=(2,2))
vectorizer.fit(train_x)
X_train=vectorizer.transform(train_x)
X_test=vectorizer.transform(test_x)

model_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
model_rf.fit(X_train, train_y)

train_score_rf=model_svm.score(X_train,train_y)
train_score_rf

score_rf=model_rf.score(X_test,test_y)
score_rf

pred_rf = model_rf.predict(X_test)

coef= model_rf.predict_proba(X_train)

importance = model_rf.feature_importances_
text = vectorizer.get_feature_names()

c=pd.DataFrame()
for i in range(len(importance)):
    a = pd.DataFrame([[text[i],importance[i]]])
    c = c.append(a)