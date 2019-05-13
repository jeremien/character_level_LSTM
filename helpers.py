import torch
import io
import numpy as np

class Helpers():

    """
    prepare data
    """

    def save_model(self, net, model_name = 'rnn_1_epoch.net'):

        checkpoint = {'n_hidden' : net.n_hidden,
                    'n_layers' : net.n_layers,
                    'state_dict':net.state_dict(),
                    'tokens': net.chars}

        with open('backup/' + model_name, 'wb') as file:
            torch.save(checkpoint, file)

    def device(self):
        if torch.cuda.is_available():
            print('cuda')
            return True
        else:
            print('cpu')
            return False

    def chars(self):
        text = self.open_file()
        text_chars = tuple(set(text))
        return text, text_chars

    def encoded(self):
        """
        encode the text and map each char to an int and vice versa
        two dict
        int2char > maps integers to char
        char2int > maps char to unique int
        """
        text, chars = self.chars()
        int2char = dict(enumerate(chars))
        char2int = {ch: ii for ii, ch in int2char.items()}
        encoded = np.array([char2int[ch] for ch in text])
        return encoded

    def one_hot_encode(self, arr, n_labels):
        """
        Initialize the the encoded array
        Fill the appropriate elements with ones
        Finally reshape it to get back to the original array
        """
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
        one_hot = one_hot.reshape((*arr.shape, n_labels))
    
        return one_hot

    def get_batches(self, arr, n_seqs, n_steps):
        '''Create a generator that returns batches of size
        n_seqs x n_steps from arr.
        
        Arguments
        ---------
        arr: Array you want to make batches from
        n_seqs: Batch size, the number of sequences per batch
        n_steps: Number of sequence steps per batch
        '''
    
        batch_size = n_seqs * n_steps
        n_batches = len(arr)//batch_size
        
        # Keep only enough characters to make full batches
        arr = arr[:n_batches * batch_size]
        
        # Reshape into n_seqs rows
        arr = arr.reshape((n_seqs, -1))
        
        for n in range(0, arr.shape[1], n_steps):
            
            # The features
            x = arr[:, n:n+n_steps]
            
            # The targets, shifted by one
            y = np.zeros_like(x)
            
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_steps]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            yield x, y


    @staticmethod
    def open_file():
        with io.open('./data/input.txt') as file:
            text = file.read()
        return text