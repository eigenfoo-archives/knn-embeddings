# kNN Text Categorizer with Word Embeddings

## Requirements

- `git`
- `g++`
- `make`
- A Python 3 interpreter

There are also a number of Python dependencies. The easiest way to install these
is to use `pip`:

```
pip install -r requirements.txt
```

## Usage:

To download or learn the word embeddings:

```
$ sh embed_glove.sh
$ sh embed_fastText.sh
$ python embed_word2vec.py
```

These three scripts will:

1. Download the [300d GloVe word
   embeddings](https://nlp.stanford.edu/projects/glove/) with `wget`, which is
   trained on 2014 Wikipedia and Gigaword 5.
2. Learn the [fastText word embeddings](https://fasttext.cc/). If fastText is
   not found in the directory, it will be automatically downloaded using `git`.
3. Learn the [word2vec word
   embeddings](https://radimrehurek.com/gensim/models/word2vec.html) using the
   python library `gensim`.

Note that these are fairly computationally intensive tasks, and required around
30 minutes to run.

To use the word embeddings in a kNN text categorizer, simply run

```
$ python knn.py
```

## Remarks

1. Stopwords didn't help. The default stopwords shipped with `nltk` were used:
   all stopwords were stripped from the embeddings and from the testing
   documents (i.e. stopwords were neither embedded nor considered during
   testing). The result was that stopwords both increased performance by at most
   1%, and decreased performance by at most 2%, depending on the corpus and the
   word embedding.
2. Just getting from word vectors to document vectors required a lot of
   experimentation: taking the maximum, minimum and the concatenation of the two
   were all found to consistently underperform taking the mean.

