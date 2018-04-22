import os
import csv
import re
from itertools import chain
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
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


def rocchio_performance(X_train, y_train, X_test, y_test):
    '''
    Given a train and test split, measure the overall accuracy,
    precision, recall, F-1 score and support of the Rocchio classifier.
    '''
    rocchio_tfidf = NearestCentroid().fit(X_train, y_train)

    predictions = rocchio_tfidf.predict(X_test)

    acc =  accuracy_score(y_test, predictions)
    prfs = np.vstack(precision_recall_fscore_support(predictions, y_test))

    print('Overall accuracy: {:f}'.format(acc))
    print('')
    print(pd.DataFrame(data=prfs,
                       index=['Precision', 'Recall', 'F-1', 'Support'],
                       columns=rocchio_tfidf.classes_))

    return acc, prfs


def get_embedding_matrix(corpus_df, embeddings, func, dim=300):
    X = np.zeros([len(corpus_df), dim])
    
    for j in range(len(corpus_df)):
        words = list(chain.from_iterable(LineSentence(corpus_df.loc[j, 'path'])))

        try:
            try:
                X[j] = func(embeddings.loc[words])
            except AttributeError:
                print('floof?')
                X[j] = func(embeddings[words])
        except:
            # FIXME out of vocab words???
            print('uh oh')
            continue
    
    return X


if __name__ == '__main__':
    '''
    print('Rocchio-tfidf')
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
        acc, prfs = rocchio_performance(tfidf_train, clf_train, tfidf_test, clf_test)
        print('')
        print('')
    print('------------------------------')


    glove = pd.read_table('embeddings/glove.6B.300d.txt',
                          delimiter=' ',
                          index_col=0,
                          header=None,
                          quoting=csv.QUOTE_NONE)


    print('Rocchio-glove (mean)')
    print('')
    for i in range(1, 4):
        train, test = get_corpus_dfs(i)

        X_train = get_embedding_matrix(train, glove, np.mean)
        X_test = get_embedding_matrix(test, glove, np.mean)
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = rocchio_performance(X_train, clf_train, X_test, clf_test)
        print('')
        print('')
    print('------------------------------')


    print('Rocchio-glove (max)')
    print('')
    for i in range(1, 4):
        train, test = get_corpus_dfs(i)

        X_train = get_embedding_matrix(train, glove, lambda x: np.amax(x, axis=0))
        X_test = get_embedding_matrix(test, glove, lambda x: np.amax(x, axis=0))
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = rocchio_performance(X_train, clf_train, X_test, clf_test)
        print('')
        print('')
    print('------------------------------')


    print('Rocchio-glove (min)')
    print('')
    for i in range(1, 4):
        train, test = get_corpus_dfs(i)

        X_train = get_embedding_matrix(train, glove, lambda x: np.amin(x, axis=0))
        X_test = get_embedding_matrix(test, glove, lambda x: np.amin(x, axis=0))
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = rocchio_performance(X_train, clf_train, X_test, clf_test)
        print('')
        print('')
    print('------------------------------')


    print('Rocchio-glove (concat max and min)')
    print('')
    for i in range(1, 4):
        train, test = get_corpus_dfs(i)


        X_train = get_embedding_matrix(train,
                                       glove,
                                       lambda x: np.hstack([np.amax(x, axis=0),
                                                            np.amin(x, axis=0)]),
                                       dim=600)
        X_test = get_embedding_matrix(test,
                                      glove,
                                      lambda x: np.hstack([np.amax(x, axis=0),
                                                           np.amin(x, axis=0)]),
                                      dim=600)
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = rocchio_performance(X_train, clf_train, X_test, clf_test)
        print('')
        print('')
    print('------------------------------')
    '''

    w2v = Word2Vec.load('embeddings/w2v.corpus1.300d')
    w2v1 = pd.DataFrame(data=w2v.wv.vectors,
                        index=w2v.wv.index2word,
                        columns=range(1, 301))
    w2v = Word2Vec.load('embeddings/w2v.corpus2.300d')
    w2v2 = pd.DataFrame(data=w2v.wv.vectors,
                        index=w2v.wv.index2word,
                        columns=range(1, 301))
    w2v = Word2Vec.load('embeddings/w2v.corpus3.300d')
    w2v3 = pd.DataFrame(data=w2v1.wv.vectors,
                        index=w2v1.wv.index2word,
                        columns=range(1, 301))
    w2v = [w2v1, w2v2, w2v3]


    print('Rocchio-word2vec (mean)')
    print('')
    for i in range(1, 4):
        print(i)
        train, test = get_corpus_dfs(i)

        X_train = get_embedding_matrix(train, w2v[i-1], np.mean)
        X_test = get_embedding_matrix(test, w2v[i-1], np.mean)
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = rocchio_performance(X_train, clf_train, X_test, clf_test)
        print('')
        print('')
    print('------------------------------')


    print('Rocchio-word2vec (max)')
    print('')
    for i in range(1, 4):
        train, test = get_corpus_dfs(i)
        
        X_train = get_embedding_matrix(train, w2v[i-1], lambda x: np.amax(x, axis=0))
        X_test = get_embedding_matrix(test, w2v[i-1], lambda x: np.amax(x, axis=0))
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = rocchio_performance(X_train, clf_train, X_test, clf_test)
        print('')
        print('')
    print('------------------------------')


    print('Rocchio-word2vec (min)')
    print('')
    for i in range(1, 4):
        train, test = get_corpus_dfs(i)

        X_train = get_embedding_matrix(train, w2v[i-1], lambda x: np.amin(x, axis=0))
        X_test = get_embedding_matrix(test, w2v[i-1], lambda x: np.amin(x, axis=0))
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = rocchio_performance(X_train, clf_train, X_test, clf_test)
        print('')
        print('')
    print('------------------------------')


    print('Rocchio-word2vec (concat max and min)')
    print('')
    for i in range(1, 4):
        train, test = get_corpus_dfs(i)

        X_train = get_embedding_matrix(train,
                                       w2v[i-1],
                                       lambda x: np.hstack([np.amax(x, axis=0),
                                                            np.amin(x, axis=0)]),
                                       dim=600)
        X_test = get_embedding_matrix(test,
                                      w2v[i-1],
                                      lambda x: np.hstack([np.amax(x, axis=0),
                                                           np.amin(x, axis=0)]),
                                      dim=600)
        clf_train = train.loc[:, 'clf']
        clf_test = test.loc[:, 'clf']

        print('Corpus {}:'.format(i))
        acc, prfs = rocchio_performance(X_train, clf_train, X_test, clf_test)
        print('')
        print('')
    print('------------------------------')
