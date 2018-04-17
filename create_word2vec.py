import os
from gensim.models.word2vec import PathLineSentences, Word2Vec


if __name__ == '__main__':

    DATA_DIRS = ['/data/corpus1/total',
                 '/data/corpus2/total',
                 '/data/corpus3/total']

    for idx, data_dir in enumerate(DATA_DIRS):
        sentences = PathLineSentences(os.getcwd() + data_dir)
        w2v = Word2Vec(sentences,
                       size=300,
                       min_count=1,
                       window=5,
                       seed=1618,
                       iter=100)
        w2v.save('w2v.corpus{}.300d'.format(idx))
