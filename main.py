from model import *
from sample import *
from figures import *

from colorama import Fore
import time, sys
from time import sleep

def delay_print(string):
    sys.stdout.write(string)
    sys.stdout.flush()
    time.sleep(.1)

def loop(text):
    for t in text:
        delay_print(t)

def main():
    if len(sys.argv) > 1:

        path = sys.argv[1]

        with open(path, 'rb') as file:
            checkpoint = torch.load(file)
            
            loaded = CharRNN(checkpoint['tokens'], 
                            n_hidden=checkpoint['n_hidden'],
                            n_layers=checkpoint['n_layers'])

            loaded.load_state_dict(checkpoint['state_dict'])

        try:
            with open('backup/text/' + time.asctime(time.localtime(time.time())) + '.txt', 'a') as file:
                while True:
                    image = figure()
                    text = '\n'.join(generate(loaded))
                    print('\n',Fore.RED, "+--------------------------------------------------------------------------------------+")
                    print('\n', image)
                    print('\n',Fore.GREEN, time.asctime(time.localtime(time.time())), '\n')
                    print(Fore.WHITE, "")
                    loop(text)
                    print('\n')
                    file.write(text)
                    sleep(20)

        except KeyboardInterrupt:
            print("stop generated text", "\n")
            file.close()
    else:
        print("Usage : python main.py backup/rnn_100_epoch_fr_256_120_4.net")

if __name__ == "__main__":
    main()
