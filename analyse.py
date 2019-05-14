import spacy
from helpers import *

nlp = spacy.load('fr_core_news_sm')


with open('data/corpus_fr/Dr Bloodmoney.txt') as file:
    text = file.read()

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
    # if ent.label_ == 'LOC':
    #     print(ent.text)