import glob, random

def figure():
    files = glob.glob('figures/' + '*.txt')
    num = random.randint(0, len(files)-1)
    file_path = files[num]
    print(file_path)
    file = open(file_path, 'r').read()
    return file

def main():
    figure()

if __name__ == "__main__":
    main()
