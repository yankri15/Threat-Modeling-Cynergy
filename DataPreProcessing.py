import io
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from PyPDF2 import PdfFileReader
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Perform basic text cleaning
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\n]", " ", text)
    text = re.sub(r"\s(?=[^a-zA-Z0-9])", "", text)
    # text = text.strip()
    return text


def process_text(text):
        # Process the text using spaCy's nlp 
        doc = nlp(text)
        # Iterate over the entities in the text
        for ent in doc.ents:
            # Print the entity text and label
            print(ent.text, ent.label_)


def tokenize_text(text):
        # Tokenize the text using spaCy's built-in tokenization method
        doc = nlp(text)
        tokens = [token.text for token in doc]
        # need to try this
        # sentences = [str(sent) for sent in doc.sents]
        return tokens

def remove_stopwords(tokens):
        # Remove stop words from the tokens
        return [token for token in tokens if token not in STOP_WORDS]

def lemmatize_tokens(tokens):
        # Lemmatize the tokens using spaCy's built-in lemmatization method
        doc = nlp(" ".join(tokens))
        lemmas = [token.lemma_ for token in doc]
        return lemmas

# Function to extract text from PDF files and preprocess the data.
def extract_text(file_path):
    with open(file_path, 'rb') as f:
        resource_manager = PDFResourceManager()
        text_stream = io.StringIO()
        laparams = LAParams()

        # Create a PDF device object that translates pages into a text stream
        device = TextConverter(resource_manager, text_stream, laparams=laparams)

        # Create a PDF interpreter object that parses pages into a document structure
        interpreter = PDFPageInterpreter(resource_manager, device)

        # Iterate over each page in the PDF file and process it
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)

        text = text_stream.getvalue()
        device.close()
        text_stream.close()

        # Preprocess text
        text = preprocess_text(text)
        # text = preprocess_text(text)

        # Use Spacy to tokenize text into sentences
        tokens = tokenize_text(text)
        
        tokens = remove_stopwords(tokens)
        tokens = lemmatize_tokens(tokens)
        return tokens

# Set up paths for train and test PDFs
train_path_1 = "Datasets/Documentation/HPE_AWB_Guide_17.20.pdf"
train_path_2 = "Datasets/Documentation/Desktop Analytics Solution - APA 7.0.pdf"
train_path_3 = "Datasets/Documentation/SIS_1131_Deployment.pdf"

# Extract text from train PDFs and combine into one text file
train_text_1 = extract_text(train_path_1)
train_text_2 = extract_text(train_path_2)
train_text_3 = extract_text(train_path_3)