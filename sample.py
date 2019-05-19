from helpers import *
from random import randint
import spacy

def generate(net):
    nlp = spacy.load('fr_core_news_sm')
    
    net.cuda()
    net.eval()

    prime = ['L','J','I']
    size = randint(500, 1000)

    result = [prime[randint(0,len(prime)-1)]]

    h = net.init_hidden(1)

    char, h = net.predict('L', h, cuda=True, top_k=5)
    result.append(char)

    for i in range(size):
        char, h = net.predict(result[-1], h, cuda=True, top_k=5)
        result.append(char)

    text = ''.join(result)

    doc = nlp(text)
    sentences = []

    for _, sentence in enumerate(doc.sents):
        phrase = str(sentence)
        sentences.append(phrase)

    sentences.pop(0)
    sentences.pop(len(sentences)-1)
    return ''.join(sentences)

def sample(net, size, prime='The', top_k=None, cuda=False):
    
    if cuda:
        net.cuda()
    else:
        net.cpu()
        
    net.eval()
    
    # first off run through the prime characters
    chars = [ch for ch in prime]

    # print(chars)
    
    h = net.init_hidden(1)
    
    for ch in prime:
        char, h = net.predict(ch, h, cuda=cuda, top_k=top_k)
        chars.append(char)
    
    # now pass in the previous char and get a new one
    for ii in range(size):
        
        char, h = net.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)
        
    return ''.join(chars)

def predict(net):

    net.eval()

    h = net.init_hidden(1)
    ch = 'L'
    net.predict(ch, h, cuda=True, top_k=5)

    return 'predict'