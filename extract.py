from PyPDF2 import PdfFileReader
import glob

def open_files(path):
    files = glob.glob(path + '*.pdf')
    return files

def text_extractor(path):
    with open(path, "rb") as file:
        pdf = PdfFileReader(file)
        info = pdf.getDocumentInfo()
        pages = pdf.getNumPages()
    
        print(pages) 
        # page = pdf.getPage(pages)
        # print(page)

        # text = page.extractText()
        # print(text)

def main():
    files = open_files('./sources/')
    # text_extractor('sources/definitions.pdf')

    print(files)

if __name__ == "__main__":
    main()