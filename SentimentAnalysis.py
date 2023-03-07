'''
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
report = classification_report(y, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
print("Classification Report:\n", df_report)
'''

import spacy
import re
import io 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

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

# Function to extract text from PDF files
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
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Use Spacy to tokenize text into sentences
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        sentences = [str(sent) for sent in doc.sents]

        return sentences



# Set up paths for train and test PDFs
train_path_1 = "Datasets/Documentation/HPE_AWB_Guide_17.20.pdf"
train_path_2 = "Datasets/Documentation/Desktop Analytics Solution - APA 7.0.pdf"
train_path_3 = "Datasets/Documentation/SIS_1131_Deployment.pdf"

# Extract text from train PDFs and combine into one text file
train_text_1 = extract_text(train_path_1)
train_text_2 = extract_text(train_path_2)
train_text_3 = extract_text(train_path_3)
train_text = train_text_1 + train_text_2 + [train_path_3]

# Write text to file
with open("train_txt.txt", "w", encoding="utf-8") as f:
    f.write('\n'.join(train_text_1 + train_text_2 + train_text_3))

# Load training data
with open('train_txt.txt', 'r', encoding='utf-8') as f:
    train_text = f.readlines()


# Create labels for training data
train_labels = [0] * len(train_text_1) + [1] * len(train_text_2) + [2] * len(train_text_3)

# Split training data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(train_text, train_labels, test_size=0.2, random_state=42)

# Vectorize training data
vectorizer = CountVectorizer(stop_words='english')
train_matrix = vectorizer.fit_transform(train_data)
val_matrix = vectorizer.transform(val_data)
# test_matrix = vectorizer.transform(test_text)

# Train model
model = MultinomialNB()
model.fit(train_matrix, train_labels)


# Test model on validation set
val_pred = model.predict(val_matrix)

# Print validation metrics
metrics = {
    'accuracy': accuracy_score(val_labels, val_pred),
    'precision': precision_score(val_labels, val_pred, average='weighted'),
    'recall': recall_score(val_labels, val_pred, average='weighted'),
    'f1_score': f1_score(val_labels, val_pred, average='weighted')
}
for metric_name, metric_value in metrics.items():
    print(f'{metric_name.capitalize()}: {metric_value:.2%}')

# Test model on test set
test_pred = model.predict(test_matrix)

# Print classification report
report = classification_report(train_labels + val_labels, list(model.predict(train_matrix + val_matrix + test_matrix)), output_dict=True)
df_report = pd.DataFrame(report).transpose()
print("Classification Report:\n", df_report)

