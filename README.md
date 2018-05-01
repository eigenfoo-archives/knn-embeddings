# kNN Text Categorizer with Word Embeddings

## Requirements

- `git`
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
$ python3 embed_word2vec.py
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
20 minutes to run on a laptop.

To run experiments on a kNN text categorizer with word embeddings:

```
$ python3 knn.py
```

## Remarks

1. Just getting from word vectors to document vectors required a lot of
   experimentation: taking the maximum, minimum and the concatenation of the two
   were all found to consistently underperform taking the mean.
2. Models based on simple averaging of word-vectors can be surprisingly good,
   (given how much information is lost in taking the average). A much more
   modern approach would be to feed the word embeddings into some kind of neural
   network.
3. Averaging vectors might be the easiest way of using word embeddings in text
   categorization but it is not the only one. It is worth trying to embed entire
   documents directly using doc2vec, or using multinomial Gaussian naive Bayes
   on the word vectors.
4. Stopwords didn't help. The default stopwords shipped with `nltk` were used:
   all stopwords were stripped from the embeddings and from the testing
   documents (i.e. stopwords were neither embedded nor considered during
   testing). The result was that stopwords both increased performance by at most
   1%, and decreased performance by at most 2%, depending on the corpus and the
   word embedding.
5. It may be worthwhile to train a corpus-specific fastText embedding: the
   pre-trained fastText embedding outperformed both the pre-trained GloVe
   embedding and the corpus-specific word2vec embedding. It appears to be a more
   robust algorithm. On the other hand, it could just be that the pre-trained
   models had far larger training corpora.
6. Conclusion: word embeddings are merely a tool. It's true that they've become
   a "secret ingredient" in most NLP systems today, but it is more important to
   worry about _how_ you use the embeddings, instead of fretting over how to
   create the embedding. In this case, kNN just might not be a very good
   algorithm.
