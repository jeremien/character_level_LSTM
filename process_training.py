import sys

from helpers import *
from model import *
from train import *

def main():
    if len(sys.argv) > 1 :
        tools = Helpers()

        text, chars = tools.chars()
        encoded = tools.encoded()

        device = tools.device()

        if 'net' in locals():
            del net

        net = CharRNN(chars, n_hidden=512, n_layers=3, drop_prob=0.7)

        print(len(text))
        print(net)

        n_seqs, n_steps = 256, 180
        model_name = sys.argv[1]

        try:
            train(net, encoded, epochs=100, n_seqs=n_seqs, n_steps=n_steps, lr=0.001, cuda=device, print_every=100)
            tools.save_model(net, model_name)
            print("model saved as {}".format(model_name))

        except KeyboardInterrupt:
            tools.save_model(net, model_name)
            print("model saved as {}".format(model_name))
    else: 
        print("Usage : python process_training.py rnn_100_epoch_fr_256_200_2_0.7.net")

if __name__ == "__main__":
    main()