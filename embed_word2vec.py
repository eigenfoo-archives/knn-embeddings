'''
Create word embeddings.
Currently only word2vec is supported.
'''

import os
import sys
from gensim.models.word2vec import PathLineSentences, Word2Vec


if __name__ == '__main__':
    try:
        embed_algo = sys.argv[1]
        dim = sys.argv[2]
        data_dir = sys.argv[3]
    except IndexError:
        msg = '''Usage: python embed.py EMBED_ALGO DIM DATA_DIR
        \tEMBED_ALGO: word2vec
        \tDIM: dimensionality of word embeddings
        \tDATA_DIR: path to directory with training data'''
        print(msg)
        sys.exit(0)

    if embed_algo == 'word2vec':
        sentences = PathLineSentences(os.getcwd() + data_dir)
        w2v = Word2Vec(sentences,
                       size=dim,
                       min_count=1,
                       window=5,
                       seed=1618,
                       iter=100)
        w2v.save('w2v.corpus{}.300d'.format(data_dir))
