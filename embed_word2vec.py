'''
Create word embeddings for corpora 1, 2 and 3 using word2vec.
'''

import os
import sys
from gensim.models.word2vec import PathLineSentences, Word2Vec


if __name__ == '__main__':
    data_dirs = ['/data/corpus1/total',
                 '/data/corpus2/total',
                 '/data/corpus3/total']

    for idx, data_dir in data_dirs:
        print('Learning word embeddings for Corpus {}...'.format(idx+1))
        sentences = PathLineSentences(os.getcwd() + data_dir)
        w2v = Word2Vec(sentences,
                       size=dim,
                       min_count=1,
                       window=5,
                       seed=1618,
                       sg=0,
                       iter=100)
        w2v.save('w2v.corpus{}.300d'.format(idx+1))
