# Download pre-trained fastText word embeddings

if ls embeddings/wiki-news-300d-1M.vec > /dev/null 2>&1 
then
    echo "fastText vectors found."
else
    echo "No fastText vectors found."
    echo
    echo "Downloading..."
    wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip
    mv wiki-news-300d-1M.vec.zip embeddings/
    cd embeddings
    echo
    echo "Unzipping..."
    unzip wiki-news-300d-1M.vec.zip
    rm wiki-news-300d-1M.vec.zip
fi

echo
echo "Success."

# Attempted to train a custom fastText embedding...
# Stopped due to time constraints.
# 
# if ls embeddings/fastText > /dev/null 2>&1 
# then
#     echo "fastText found."
# else
#     echo "fastText not found."
#     echo
#     echo "Downloading..."
#     git clone https://github.com/facebookresearch/fastText.git
#     mv fastText/ embeddings/
#     cd embeddings/fastText/
#     echo
#     echo "Compiling..."
#     make
#     cd ../../
# fi
# 
# echo
# echo "Learning embeddings for Corpus 1..."
# ./embeddings/fastText/fasttext cbow -input data/corpus1/total/total.txt \
#     -output fastText.corpus1.300d -dim 300 -minCount 1 -epoch 100
# 
# echo
# echo "Learning embeddings for Corpus 2..."
# ./embeddings/fastText/fasttext cbow -input data/corpus2/total/total.txt \
#     -output fastText.corpus2.300d -dim 300 -minCount 1 -epoch 100
# 
# echo
# echo "Learning embeddings for Corpus 3..."
# ./embeddings/fastText/fasttext cbow -input data/corpus3/total/total.txt \
#     -output fastText.corpus3.300d -dim 300 -minCount 1 -epoch 100
# 
# mv fastText.* embeddings/
# 
# echo
# echo "Success."
