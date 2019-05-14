
from helpers import *
from model import *
from train import *

tools = Helpers()

text, chars = tools.chars()
encoded = tools.encoded()

device = tools.device()

if 'net' in locals():
    del net

net = CharRNN(chars, n_hidden=512, n_layers=4)

print(len(text))
print(net)

n_seqs, n_steps = 256, 150
model_name = 'rnn_1_epoch.net'

try:
    train(net, encoded, epochs=50, n_seqs=n_seqs, n_steps=n_steps, lr=0.001, cuda=device, print_every=10)
   
    tools.save_model(net, model_name)

except KeyboardInterrupt:

    tools.save_model(net, model_name)