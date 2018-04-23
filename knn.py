import os
import csv
import re
from itertools import chain
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from gensim.models.word2vec import Word2Vec, LineSentence


def get_corpus_dfs(n):
    if n == 1:
        s = 'corpus1'
    elif n == 2:
        s = 'split_corpus2'
    elif n == 3:
        s = 'split_corpus3'

    train = pd.read_csv('data/{}_train.labels'.format(s),
                        delim_whitespace=True,
                        names=['path', 'clf'])
    test = pd.read_csv('data/{}_test.labels'.format(s),
                       delim_whitespace=True,
                       names=['path', 'clf'])

    train.loc[:, 'path'] = train.loc[:, 'path'].map(lambda s: 'data' + s[1:])
    test.loc[:, 'path'] = test.loc[:, 'path'].map(lambda s: 'data' + s[1:])

    return train, test


def knn_performance(X_train, y_train, X_test, y_test):
    '''
    Given a train and test split, measure the overall accuracy,
    precision, recall, F-1 score and support of the kNN classifier.
    '''
    knn = KNeighborsClassifier(n_neighbors=5,
                               weights='distance').fit(X_train, y_train)

    predictions = knn.predict(X_test)

    acc =  accuracy_score(y_test, predictions)
    prfs = np.vstack(precision_recall_fscore_support(predictions, y_test))

    print('Overall accuracy: {:f}'.format(acc))
    print('')
    print(pd.DataFrame(data=prfs,
                       index=['Precision', 'Recall', 'F-1', 'Support'],
                       columns=knn.classes_))

    return acc, prfs

'''
with open('stopwords.txt', 'r') as f:
    stops = f.read().split()
'''


def get_embedding_matrix(corpus_df, embeddings, func, stopwords=[], dim=300):
    X = np.zeros([len(corpus_df), dim])
    
    for j in range(len(corpus_df)):
        words = list(chain.from_iterable(LineSentence(corpus_df.loc[j, 'path'])))
        words = [w for w in words if w not in stopwords]
        X[j] = func(embeddings.loc[words])
    
    return X


if __name__ == '__main__':
    print('tfidf')
    print('')
    for i in range(1, 4):
        train, test = get_corpus_dfs(i)

        vec = TfidfVectorizer(input='filename',
                              strip_accents='unicode',
                              stop_words='english',
                              max_df=0.90,
                              min_df=2,
                              norm='l2')

        tfidf_train = vec.fit_transform(train.loc[:, 'path'])
        tfidf_test = vec.transform(test.loc[:, 'path'])
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = knn_performance(tfidf_train, clf_train, tfidf_test, clf_test)
        print('')
        print('')
    print('------------------------------')


    glove = pd.read_table('embeddings/glove.6B.300d.txt',
                          delimiter=' ',
                          index_col=0,
                          header=None,
                          quoting=csv.QUOTE_NONE)


    print('GloVe')
    print('')
    for i in range(1, 4):
        train, test = get_corpus_dfs(i)

        X_train = get_embedding_matrix(train, glove, np.mean)
        X_test = get_embedding_matrix(test, glove, np.mean)
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = knn_performance(X_train, clf_train, X_test, clf_test)
        print('')
        print('')
    print('------------------------------')


    w2v = Word2Vec.load('embeddings/w2v.corpus1.300d')
    w2v1 = pd.DataFrame(data=w2v.wv.vectors,
                        index=w2v.wv.index2word,
                        columns=range(1, 301))
    w2v = Word2Vec.load('embeddings/w2v.corpus2.300d')
    w2v2 = pd.DataFrame(data=w2v.wv.vectors,
                        index=w2v.wv.index2word,
                        columns=range(1, 301))
    w2v = Word2Vec.load('embeddings/w2v.corpus3.300d')
    w2v3 = pd.DataFrame(data=w2v.wv.vectors,
                        index=w2v.wv.index2word,
                        columns=range(1, 301))
    w2v = [w2v1, w2v2, w2v3]


    print('word2vec')
    print('')
    for i in range(1, 4):
        train, test = get_corpus_dfs(i)

        X_train = get_embedding_matrix(train, w2v[i-1], np.mean)
        X_test = get_embedding_matrix(test, w2v[i-1], np.mean)
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = knn_performance(X_train, clf_train, X_test, clf_test)
        print('')
        print('')
    print('------------------------------')

    # FIXME now do fastText!
