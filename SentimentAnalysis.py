import spacy
import pandas as pd
import io
import re
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Open PDF file
with open('Datasets/Documentation/Whatfix Implementation and Security Document.pdf', 'rb') as pdf_file:
    # Extract text from PDF file
    resource_manager = PDFResourceManager()
    text_stream = io.StringIO()
    device = TextConverter(resource_manager, text_stream, laparams=LAParams())
    interpreter = PDFPageInterpreter(resource_manager, device)

    for page in PDFPage.get_pages(pdf_file):
        interpreter.process_page(page)

    text = text_stream.getvalue()
    text_stream.close()
    
    # Preprocess text
    text = re.sub(r'\s+', ' ', text) # remove extra spaces
    text = text.strip() # remove leading and trailing whitespace
    
    # Use Spacy to tokenize text into sentences
    doc = nlp(text)
    sentences = [str(sent) for sent in doc.sents]

# Write sentences to CSV file
df = pd.DataFrame(sentences, columns=['text'])
df.to_csv('sentences.csv', index=False)
