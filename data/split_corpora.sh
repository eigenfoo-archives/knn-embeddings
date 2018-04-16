# Randomly segment the files corpus2_train.labels and corpus3_train.labels so as
# to create a similar file structure to corpus1, with 1/3 train-test split.
#
# Run on a Mac OSX (note the use of gshuf, ghead and gtail). Simply remove the
# g's to run on any other OS.

gshuf corpus2_train.labels > split_corpus2_train.labels
gtail -n 298 split_corpus2_train.labels > split_corpus2_test.labels
ghead -n -298 split_corpus2_train.labels > temp.txt
mv temp.txt split_corpus2_train.labels
cp split_corpus2_test.labels split_corpus2_test.list
sed 's/ .$//' split_corpus2_test.list > temp.txt
mv temp.txt split_corpus2_test.list

gshuf corpus3_train.labels > split_corpus3_train.labels
gtail -n 318 split_corpus3_train.labels > split_corpus3_test.labels
ghead -n -318 split_corpus3_train.labels > temp.txt
mv temp.txt split_corpus3_train.labels
cp split_corpus3_test.labels split_corpus3_test.list
sed 's/ ...$//' split_corpus3_test.list > temp.txt
mv temp.txt split_corpus3_test.list
