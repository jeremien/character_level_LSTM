
from helpers import *
from model import *
from train import *

tools = Helpers()

_, chars = tools.chars()
encoded = tools.encoded()

device = tools.device()

if 'net' in locals():
    del net

net = CharRNN(chars, n_hidden=512, n_layers=2)

print(net)

n_seqs, n_steps = 128, 100

train(net, encoded, epochs=5, n_seqs=n_seqs, n_steps=n_steps, lr=0.001, cuda=device, print_every=10)

model_name = 'rnn_1_epoch.net'

checkpoint = {'n_hidden' : net.n_hidden,
              'n_layers' : net.n_layers,
              'state_dict':net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as file:
    torch.save(checkpoint, file)