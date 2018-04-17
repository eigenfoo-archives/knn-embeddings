import os
from gensim.models.word2vec import PathLineSentences, Word2Vec


if __name__ == '__main__':
    sentences = PathLineSentences(os.getcwd() + '/data/corpus1/train')
    w2v = Word2Vec(sentences,
                   size=100,
                   window=5,
                   seed=1618)
    w2v.save('w2v.100d.txt')
