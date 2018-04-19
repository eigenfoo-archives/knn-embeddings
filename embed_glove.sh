# Obtain general-purpose word embeddings from GloVe.

if ls embeddings/glove.6B.300d.txt > /dev/null 2>&1 
then
    echo "GloVe vectors found."
else
    wget http://nlp.stanford.edu/data/glove.6B.zip
    mv glove.6B.zip embeddings/
    cd embeddings
    unzip glove.6B.zip
fi

echo
echo "Success."
