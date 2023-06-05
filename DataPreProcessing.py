import io
import re
import spacy
import nltk
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3000000  
nltk.download('punkt')

# List of tags I got from "Cynergy" in order to detect potential cyber security threats in documentations files.
tags = ['register', 'signup', 'sign-up', 'password', 'recovery', 'chat', 'bot',
        'cart', 'bag', 'basket', 'profile', 'account', 'user', 'settings', 'coupon', 
        'promotion', 'wp-login', 'login', 'log-in', 'signin', 'sign-in', 'signout', 
        'sign-out', 'logout', 'log-out', 'contact', 'search', 'href']


# preprocess the test by normalizing it.
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\n\.]", " ", text)
    sentences = text.split("\n")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    sentences = [re.sub(r"^\s+|\s+$", "", sentence) for sentence in sentences]
    sentences = [s for s in sentences if len(s) >= 5]
    text = "\n".join(sentences)
    return text


# Tokenize the text using spaCy's built-in tokenization method
def tokenize_text(text):
        doc = nlp(text)
        sentences = [str(sent) for sent in doc.sents]
        return sentences


# Remove stop words from the tokens
def remove_stopwords(tokens):
        return [token for token in tokens if token not in STOP_WORDS]


# Lemmatize the tokens using spaCy's built-in lemmatization method
def lemmatize_sentences(sentences):
    lemmatized_sentences = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        doc = nlp(" ".join(tokens))
        lemmas = [token.lemma_ for token in doc]
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
        tokens = tokenize_text(text)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize_sentences(tokens)
        return tokens

# Label each sentence by determaining if the sentence contains a potential threat
# Need to upload working chnages to here (problem with pc)
def label_threats(sentences, tags):
    df = pd.DataFrame(columns=["sentence", "threat"])
    for sentence in sentences:
        if any(tag in sentence for tag in tags):
            #label it as a positive match (1)
            new_row = pd.DataFrame({"sentence": [sentence], "threat": [1]})
        else:
            #label it as a negative match (0)
            new_row = pd.DataFrame({"sentence": [sentence], "threat": [0]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df


# Count number of sentences that were classified as threats and non - threats
def count_threats(df):
    num_threats = df[df['threat'] == 1].shape[0]
    num_non_threats = df[df['threat'] == 0].shape[0]
    return num_threats, num_non_threats


# Clean the dataframe by removing any noise left after initial prprocessing phase
def clean_dataframe(df):
    df["sentence"] = df["sentence"].apply(lambda x: x.replace(".", ""))
    # Remove sentences containing only one word
    df = df[df["sentence"].str.split().apply(len) > 1]
    df = df.reset_index(drop=True)
    return df


# Set up paths for train and test PDFs
train_path_1 = "Datasets/Documentation/CyberArk-and-Workfusion-IA2017Sunbird-integration-v3.pdf"
train_path_2 = "Datasets/Documentation/Desktop Analytics Solution - APA 7.0.pdf"
train_path_3 = "Datasets/Documentation/HPE_AWB_Guide_17.20.pdf"
train_path_4 = "Datasets/Documentation/SIS_1131_Deployment.pdf"
train_path_5 = "Datasets/Documentation/Whatfix Implementation and Security Document.pdf"
train_path_6 = "Datasets/Documentation/BSM_Integrations_HPOM.pdf"
train_path_7 = "Datasets/Documentation/fortiweb-v6.1.0-admin-guide.pdf"
train_path_8 = "Datasets/Documentation/db2z_12_adminbook.pdf"
train_path_9 = "Datasets/Documentation/pingfederate-110.pdf"
train_path_10 = "Datasets/Documentation/sg246098.pdf"

# Extract text from train PDFs and combine into one text file
processed_text_1 = extract_text(train_path_1)
processed_text_2 = extract_text(train_path_2)
processed_text_3 = extract_text(train_path_3)
processed_text_4 = extract_text(train_path_4)
processed_text_5 = extract_text(train_path_5)
processed_text_6 = extract_text(train_path_6)
processed_text_7 = extract_text(train_path_7)
processed_text_8 = extract_text(train_path_8)
processed_text_9 = extract_text(train_path_9)
processed_text_10 = extract_text(train_path_10)

processed_sentences = processed_text_1 + processed_text_2 + processed_text_3 + processed_text_4 + processed_text_5 + processed_text_6 + processed_text_7 + processed_text_8 + processed_text_9 + processed_text_10
labeled_df = label_threats(processed_sentences, tags)
labeled_df = clean_dataframe(labeled_df)

num_threats, num_non_threats = count_threats(labeled_df)
print(num_threats)
print(num_non_threats)

labeled_df.to_csv("data.csv", index=False)
