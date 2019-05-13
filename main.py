from model import *
from sample import *

with open('rnn_1_epoch.net', 'rb') as file:
    checkpoint = torch.load(file)
    
loaded = CharRNN(checkpoint['tokens'], 
                 n_hidden=checkpoint['n_hidden'],
                 n_layers=checkpoint['n_layers'])

loaded.load_state_dict(checkpoint['state_dict'])

sample(loaded, 100, prime="I", top_k=5, cuda=True)