#!/usr/bin/env python3.5.6

# import sys

# print(sys.version)
# for p in sys.path:
#     print(p)
    

from model import *
from sample import *
from colorama import Fore
import time, sys, random, glob

def delay_print(string):
    # sys.stdout.write(string)
    # sys.stdout.flush()
    print(string)
    random_num = random.uniform(0.7, 0.001)
    # random_num = random.random()
    # print(random_num)
    time.sleep(random_num)

def loop(text):
    for t in text:
        delay_print(t)

def figure():
    files = glob.glob('/home/administrateur/character_level_LSTM/figures/' + '*.txt')
    num = random.randint(0, len(files)-1)
    file_path = files[num]
    file = open(file_path, 'r').read()
    return file

def main():

    path = "/home/administrateur/character_level_LSTM/backup/rnn_200_256_140_4_0.6.net"

    with open(path, 'rb') as file:
        checkpoint = torch.load(file)
        
        loaded = CharRNN(checkpoint['tokens'], 
                        n_hidden=checkpoint['n_hidden'],
                        n_layers=checkpoint['n_layers'])

        loaded.load_state_dict(checkpoint['state_dict'])
    image = figure()
    date = time.asctime(time.localtime(time.time()))
    text = '\n'.join(generate(loaded))

    print('\n',Fore.RED, "")
    print('\n', image)
    print('\n',Fore.GREEN, date)
    print(Fore.WHITE, "")
    print(text)


if __name__ == "__main__":
    main()
