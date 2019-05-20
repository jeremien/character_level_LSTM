from flask import Flask
from model import *
from sample import *
import sys, time, json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]

        with open(path, 'rb') as file:
                checkpoint = torch.load(file)
                
                loaded = CharRNN(checkpoint['tokens'], 
                                n_hidden=checkpoint['n_hidden'],
                                n_layers=checkpoint['n_layers'])

                loaded.load_state_dict(checkpoint['state_dict'])
        
        date = time.asctime(time.localtime(time.time()))
        text = generate(loaded)
        
        data = {
            'date' : date,
            'text' : text
        }
        return json.dumps(data) 
    else:
        print("Usage : python server.py backup/rnn_100_epoch_fr_256_120_4.net")

if __name__ == '__main__':
    app.run(debug = True)