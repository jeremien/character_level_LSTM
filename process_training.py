
from helpers import *
from model import *
from train import *

tools = Helpers()

text, chars = tools.chars()
encoded = tools.encoded()

device = tools.device()

if 'net' in locals():
    del net

net = CharRNN(chars, n_hidden=512, n_layers=3, drop_prob=0.7)

print(len(text))
print(net)

n_seqs, n_steps = 256, 200
model_name = 'rnn_100_epoch_fr_256_200_3_0.7.net'

try:
    train(net, encoded, epochs=100, n_seqs=n_seqs, n_steps=n_steps, lr=0.001, cuda=device, print_every=100)
    tools.save_model(net, model_name)

except KeyboardInterrupt:
    tools.save_model(net, model_name)
    print("model saved")