import os
import spacy
nlp = spacy.load('en', disable=['tagger', 'ner'])

if __name__ == '__main__':
    dir_name = 'corpus3/total'
    for filename in os.listdir(dir_name):
        with open(os.path.join(os.getcwd(), dir_name, filename), 'r+') as f:
            s = f.read()

            doc = nlp(s)
            sentences = list(doc.sents)

            f.seek(0)
            f.truncate()

            for sentence in sentences:
                print(''.join(sentence.text.splitlines()), file=f)
