import pdfplumber
import spacy


def sentenceExtraction(pdf):
    nlp = spacy.load("en_core_web_sm")

    with pdfplumber.open("PDFs\A Brief Overview of the History of Computers.pdf") as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()


    text = text.lower()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]


