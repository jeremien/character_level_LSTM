from PyPDF2 import PdfFileReader
import glob

def open_files(path):
    files = glob.glob(path + '*.pdf')
    return files

def text_extractor(path):
        result = []
        with open(path, "rb") as file:
                pdf = PdfFileReader(file)
                info = pdf.getDocumentInfo()
                pages = pdf.getNumPages()
                for p in range(pages):
                        page = pdf.getPage(p)
                        text = page.extractText()
                        result.append(text)
        print(result)

def main():
    files = open_files('./sources/neuroscience/')

    for file in files:
        text_extractor(file)

if __name__ == "__main__":
    main()