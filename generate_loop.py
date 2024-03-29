from model import *
from sample import *
from colorama import Fore
import time, sys, random, glob, os

dirpath = os.getcwd()

def delay_print(string):
    sys.stdout.write(string)
    sys.stdout.flush()
    random_num = random.uniform(0.7, 0.001)
    time.sleep(random_num)

def loop(text):
    for t in text:
        delay_print(t)

def figure():
    files = glob.glob(dirpath + '/figures/' + '*.txt')
    num = random.randint(0, len(files)-1)
    file_path = files[num]
    file = open(file_path, 'r').read()
    return file

def main():
    path = dirpath + '/backup/rnn_150_en_256_130_4_0.5.net'
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
        print(image)
        print('\n',Fore.GREEN, date)
        print(Fore.WHITE, "")
        loop(text)
        print('\n')

for i in range(2678460):
    if i % 2 == 0:
        print(Fore.BLUE, """
           _  __       __    __    _                             
          / |/ / ___  / /_  / /   (_)  ___   ___ _               
         /    / / _ \/ __/ / _ \ / /  / _ \ / _ `/               
        /_/|_/  \___/\__/ /_//_//_/_ /_//_/ \_, /                
         | | /| / /  (_)  / / ___/ /       /___/                 
         | |/ |/ /  / /  / / / _  /                              
         |__/|__/  /_/  /_/  \_,_/                               
          (_)  ___                                               
         / /  / _ \                                              
        /_/__/_//_/          __    _               __            
          / _ \ ___ _  ____ / /_  (_) ____ __ __  / / ___ _  ____
         / ___// _ `/ / __// __/ / / / __// // / / / / _ `/ / __/
        /_/    \_,_/ /_/   \__/ /_/  \__/ \_,_/ /_/  \_,_/ /_/  
        
        """)
    print(Fore.GREEN,"Text n°{}".format(i+1))
    main()
    time.sleep(60)
