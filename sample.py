from helpers import *

def sample(net, size, prime='The', top_k=None, cuda=False):
    
    if cuda:
        net.cuda()
    else:
        net.cpu()
        
    net.eval()
    
    # first off run through the prime characters
    chars = [ch for ch in prime]
    
    h = net.init_hidden(1)
    
    for ch in prime:
        char, h = net.predict(ch, h, cuda=cuda, top_k=top_k)
        
    chars.append(char)
    
    # now pass in the previous char and get a new one
    for ii in range(size):
        
        char, h = net.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)
        
    return ''.join(chars)