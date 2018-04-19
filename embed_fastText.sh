if ls embeddings/fastText > /dev/null 2>&1 
then
    echo "fastText found."
else
    git clone https://github.com/facebookresearch/fastText.git
    mv fastText/ embeddings/
    cd embeddings/fastText/
    make
fi


