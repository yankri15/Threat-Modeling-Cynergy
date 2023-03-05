import spacy
import pandas as pd
import io
import re
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Define the tags and their corresponding categories
tags = [['register', 'signup', 'sign-up'],
        ['password', 'recovery'],
        ['chat', 'bot'],
        ['cart', 'bag', 'basket'],
        ['profile', 'account', 'user', 'settings'],
        ['coupon', 'promotion'],
        ['wp-login', 'login', 'log-in', 'signin', 'sign-in', 'signout', 'sign-out', 'logout', 'log-out'],
        ['contact'],
        ['search'],
        ['href']]

tagsPerCategory = {
    'RegisterForm': tags[0],
    'RstPss': tags[1],
    'ChatBot': tags[2],
    'BagCart': tags[3],
    'AccountSettings': tags[4],
    'CouponPromotion': tags[5],
    'LoginLogout': tags[6],
    'ContactUs': tags[7],
    'Search': tags[8],
    'Links': tags[9]
}

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
    text = text.lower()
    # text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r'\s+', ' ', text) 
    text = text.strip() 
    
    # Use Spacy to tokenize text into sentences
    doc = nlp(text)
    sentences = [str(sent) for sent in doc.sents]

# Write sentences to CSV file
df = pd.DataFrame(sentences, columns=['text'])
df.to_csv('sentences.csv', index=False)

# Load sentences from CSV file
df = pd.read_csv('sentences.csv')

# Add category labels to DataFrame
def get_category_tags(tagsPerCategory, tags):
    for category, category_tags in tagsPerCategory.items():
        for tag in category_tags:
            if tag in tags:
                return category
    return None

df['label'] = df['text'].apply(lambda x: get_category_tags(tagsPerCategory, x))

# Remove rows with missing labels
df = df.dropna(subset=['label'])

# Preprocess sentences and tokenize with Spacy
docs = [nlp(sent.lower()) for sent in df['text']]

# Define X and y
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([doc.text for doc in docs])
y = df['label']

# Train Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

# Make predictions on test set
y_pred = clf.predict(X)

# Print classification report
print(classification_report(y, y_pred))
