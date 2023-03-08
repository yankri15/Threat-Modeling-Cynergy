import io
import re
import spacy
import nltk
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from PyPDF2 import PdfFileReader
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
tags = ['register', 'signup', 'sign-up', 'password', 'recovery', 'chat', 'bot',
        'cart', 'bag', 'basket', 'profile', 'account', 'user', 'settings', 'coupon', 
        'promotion', 'wp-login', 'login', 'log-in', 'signin', 'sign-in', 'signout', 
        'sign-out', 'logout', 'log-out', 'contact', 'search', 'href']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\n\.]", " ", text)
    sentences = text.split("\n")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    sentences = [re.sub(r"^\s+|\s+$", "", sentence) for sentence in sentences]
    sentences = [s for s in sentences if len(s) >= 5]
    text = "\n".join(sentences)
    return text


def tokenize_text(text):
        # Tokenize the text using spaCy's built-in tokenization method
        doc = nlp(text)
        # tokens = [token.text for token in doc]
        # need to try this
        sentences = [str(sent) for sent in doc.sents]
        return sentences

def remove_stopwords(tokens):
        # Remove stop words from the tokens
        return [token for token in tokens if token not in STOP_WORDS]

def lemmatize_sentences(sentences):
    lemmatized_sentences = []
    for sentence in sentences:
        # Tokenize the sentence
        tokens = nltk.word_tokenize(sentence)
        # Lemmatize the tokens using spaCy's built-in lemmatization method
        doc = nlp(" ".join(tokens))
        lemmas = [token.lemma_ for token in doc]
        # Join the lemmas back into a sentence
        lemmatized_sentence = " ".join(lemmas)
        lemmatized_sentences.append(lemmatized_sentence)
    return lemmatized_sentences


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

        # Use Spacy to tokenize text into sentences
        tokens = tokenize_text(text)
        
        tokens = remove_stopwords(tokens)
        tokens = lemmatize_sentences(tokens)
        return tokens

def label_threats(sentences, tags):
    # Create an empty dataframe with columns "sentence" and "threat"
    df = pd.DataFrame(columns=["sentence", "threat"])
    # Loop over each sentence and check if it contains any of the tags
    for sentence in sentences:
        # Check if the sentence contains any of the tags
        if any(tag in sentence for tag in tags):
            # If it does, label it as a positive match (1)
            new_row = pd.DataFrame({"sentence": [sentence], "threat": [1]})
        else:
            # If it doesn't, label it as a negative match (0)
            new_row = pd.DataFrame({"sentence": [sentence], "threat": [0]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df


# Set up paths for train and test PDFs
train_path_1 = "Datasets/Documentation/HPE_AWB_Guide_17.20.pdf"
train_path_2 = "Datasets/Documentation/Desktop Analytics Solution - APA 7.0.pdf"
train_path_3 = "Datasets/Documentation/SIS_1131_Deployment.pdf"

# Extract text from train PDFs and combine into one text file
processed_text_1 = extract_text(train_path_1)
processed_text_2 = extract_text(train_path_2)
processed_text_3 = extract_text(train_path_3)

processed_sentences = processed_text_1 + processed_text_2 + processed_text_3
labeled_df = label_threats(processed_sentences, tags)

labeled_df.to_csv("data.csv", index=False)