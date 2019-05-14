import unicodedata
import os, sys, glob, re, pickle, string
from progress.bar import Bar


class FormatData():
    
    """
    merge corpus of text files into one input for RNN
    args : 
        path (str) : path of the corpus
    """
    
    def __init__(self, path):
        self.path = path

    def open_files(self):
        files = glob.glob(self.path + '*.txt')
        # print(files)
        with open('./data/input.txt', 'w') as outfile:
            if files:
                for file in files:
                    # print(file)
                    with open(file) as infile:
                        for line in infile:
                            if not line.strip(): continue
                            line = re.sub('^\d*', '', line)
                            line = re.sub('[«»]', '', line)
                            line = re.sub('[—] ','', line)
                            line = re.sub('[–]', '', line)
                            line = re.sub(' [—] ','', line)
                            line = re.sub('  [–] ', '', line)
                            line = re.sub('[…]','.', line)
                            # line = re.sub(' \. ', '.', line)
                            # line = re.sub('.$', ' ', line)
                            # line = re.sub('[\n\t\r]', ' ', line)
                            line = re.sub('[’]', "'", line)
                            # line = self.unicode_to_ascii(line)
                            line = re.sub('[.]{2,}', ' ', line)
                            line = re.sub('\s+', ' ', line)
                            # line = line.lower()
                            line = re.sub('[ ]{2,}', ' ', line)
                            # line = re.sub('^\s', '', line)
                            line = re.sub('[*] ', '', line)
                            line = re.sub(' [,] ', ', ', line)
                            outfile.write(line)
                        print("save file {}".format(file))    

            else:
                print("no files")


    def unicode_to_ascii(self, text):
        try:
            all_letters = string.ascii_letters + " àéèêâ?!,;-\"':."
            # all_letters = string.ascii_letters + " '"
            return ''.join(
                c for c in unicodedata.normalize('NFKD', text)
                if unicodedata.category(c) != 'Mn'
                and c in all_letters
            )
        except:
            print("error when parsing from unicode to ascii")

def main(*args):
    data = FormatData(args[0])
    data.open_files()
    print("files processed")

if __name__ == "__main__":
    main(str(sys.argv[1]))