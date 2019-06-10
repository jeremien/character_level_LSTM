from model import *
from sample import *
from colorama import Fore
import time, sys, random, glob

def delay_print(string):
    sys.stdout.write(string)
    sys.stdout.flush()
    random_num = random.uniform(0.7, 0.001)
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
    path = "/home/administrateur/character_level_LSTM/backup/rnn_200_256_140_4_0.6_dick.net"
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
        loop(text)
        print('\n')

for i in range(2678460):
    print(Fore.GREEN,"itération <{}>".format(i))
    main()
    time.sleep(60)
# if __name__ == "__main__":

    # while True:
    #     main()
    #     time.sleep(20)
    # except KeyboardInterrupt:
    #     print("stop generated text", "\n")

