from model import *
from sample import *
from colorama import Fore, Back
import time
from time import sleep

with open('backup/rnn_50_epoch_fr.net', 'rb') as file:
    checkpoint = torch.load(file)
    
loaded = CharRNN(checkpoint['tokens'], 
                 n_hidden=checkpoint['n_hidden'],
                 n_layers=checkpoint['n_layers'])

loaded.load_state_dict(checkpoint['state_dict'])

def main():
    try:
        while True:
            # print('\n',Back.BLUE)
            print('\n',Fore.RED, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print('\n',Fore.GREEN, time.asctime(time.localtime(time.time())), '\n')
            print(Fore.WHITE, generate(loaded))
            print('\n\n\n')
            sleep(20)

    except KeyboardInterrupt:
        print("stop generated text", "\n")

if __name__ == "__main__":
    main()
