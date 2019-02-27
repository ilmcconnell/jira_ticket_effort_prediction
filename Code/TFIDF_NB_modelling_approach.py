# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 21:25:07 2016
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from sklearn import metrics


#import data and inspect 
path = ".../Data/"
filename = '20161201JiraTicketExport.csv'
df = pd.DataFrame()
df = pd.read_csv(path+filename,parse_dates=[10,11,12,13,14,15,16,17],infer_datetime_format=True)
print df.head()
print df.info()

#add basic features
df['summary_len'] = [len(s) for s in df['Summary']]
df['word_Ct'] = [int(s.count(' ')) + 1 for s in df['Summary']]

#examine outliers
print df[ df['summary_len'] > 218][:]['Summary']
print df.describe()  #.summary()

# feature_engineering and data conversion
df['Class'] = [0 if row<=1 else 1 for row in df['EffortInHours'] ]
df['SummaryLength'] = [len(row) for row in df['Summary']]
df['SummaryDecode'] = [row.decode('windows-1252').strip() for row in df['Summary']]
df['SummaryDecode'].head(20)
df.hist('SummaryLength', bins=20, figsize=[12,8] )
df.head()


#data exploration
plt.style.use('ggplot')

plt.figure(1)
df.boxplot(column='EffortInHours', figsize=[3, 6]) 
plt.savefig('../viz/effortTotalBoxPlot')

plt.figure(2)
df[df['EffortInHours'] <= 20].hist(column='EffortInHours', xrot=90, sharex=True, sharey=True, figsize=[9, 6], bins=20)
plt.savefig('../viz/effortTotalPerTicketNoFlyersHist')



plt.figure(3)
df.boxplot(column='EffortInHours', figsize=[12, 8], showfliers=False) 
plt.savefig('../viz/effortTotalNoFliersBoxPlot')

plt.figure(4)
df[df['EffortInHours'] <= 20].boxplot(column='EffortInHours', figsize=[12, 8])
plt.savefig('../viz/effortTotalTwentyOrLessBoxPlot')

plt.figure(5)
df.hist(column='Class', figsize=[12, 8], bins=2 )
plt.savefig('../viz/effortClassBar')


# features data viz
plt.figure(6)
df[(df['EffortInHours'] <= 20) & (df['summary_len'] <= 200)].boxplot(column='summary_len', figsize=[12, 8]) #, return_type='dict')
plt.savefig('../viz/SummaryLenFor20HoursOrLessBoxPlot')

plt.figure(7)
df[(df['EffortInHours'] <= 20) & (df['summary_len'] <= 200)].hist(column='summary_len', xrot=90, sharex=True, sharey=True, figsize=[12, 8], bins=20)
plt.savefig('../viz/SummaryLenFor20HoursOrLessHistogram')

plt.figure(7)
df[(df['EffortInHours'] <= 20) & (df['word_Ct'] <= 50)].boxplot(column='word_Ct', figsize=[12, 8]) #, return_type='dict')
plt.savefig('../viz/WordCtFor20HoursOrLessBoxPlot')

plt.figure(9)
df[(df['EffortInHours'] <= 20) & (df['word_Ct'] <= 40)].hist(column='word_Ct', figsize=[12, 8], bins=40)
# xrot = 90, sharex = True, sharey = True,
plt.savefig('../viz/WordCtFor20HoursOrLessHistogram')

#correlation??
plt.figure(10)
# df['NotClass'] = []
df[(df['EffortInHours'] <= 20) & (df['word_Ct'] <= 40)].plot.scatter(x='word_Ct', y='EffortInHours', c='Class', colorbar=False, figsize=[12,8], xlim=[0,40], ylim=[0,20])
# xrot = 90, sharex = True, sharey = True,
plt.savefig('../viz/EffortVsWordCt')

plt.figure(11)
# df['NotClass'] = []
df[(df['EffortInHours'] <= 20) & (df['word_Ct'] <= 40)].plot.scatter(x='word_Ct', y='EffortInHours', c='Class', colorbar=False, figsize=[9,6], xlim=[0,40], ylim=[0,20])
# xrot = 90, sharex = True, sharey = True,
plt.savefig('../viz/EffortVsWordCt96')

plt.figure(12)
# df['NotClass'] = []
df[(df['EffortInHours'] <= 20) & (df['word_Ct'] <= 40)].plot.scatter(x='word_Ct', y='EffortInHours', c='Class', colorbar=False, figsize=[7,5], xlim=[0,40], ylim=[0,20])
# xrot = 90, sharex = True, sharey = True,
plt.savefig('../viz/EffortVsWordCt75')

# output rows with certain word counts - examine strange feature in the data
twelves = df[df['word_Ct'] == 12].sort_values('Summary').to_csv(path+'twelves.csv')


#TFIDF Vectorization

# text to vectorize
X = df['SummaryDecode']
# labels for estimator
y = df['Class']

#instantiate vectorizer
tfidfVect = TfidfVectorizer(
                            decode_error='ignore'
                            , ngram_range=(1,100)
                            , max_features=46 #SQRT 2160
                            , min_df = 100
                            , stop_words='english')
                             #try to limit max frequency of a feature, and use a pipeline? in a loop - measure tweaks.

#break out data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y)


# transform text of training and test data based on training data
X_train_dtmTfidf = tfidfVect.fit_transform(X_train)
X_test_dtmTfidf = tfidfVect.transform(X_test)


# predict ticket class based on tokenization
nb = MultinomialNB()
nb.fit(X_train_dtmTfidf, y_train)
y_pred_class = nb.predict(X_test_dtmTfidf)

print("classification_report:")
print(classification_report(y_test, y_pred_class))

#accuracy
print 'tfidf accuracy: %s' % (metrics.accuracy_score(y_test, y_pred_class))

#null accuracy
nullacc = (sum(df['Class']) * 1.0) / (len(df['Class']) * 1.0 )
print "null accuracy: %s" % str(nullacc)

# confusion matrix
print(confusion_matrix(y_test, y_pred_class, labels=["0","1"], sample_weight=None))
