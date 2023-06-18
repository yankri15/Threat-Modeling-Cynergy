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
keywords = [['register', 'signup', 'sign-up', 'join'], ['password', 'recovery', 'reset', 'forgot'],
            ['chat', 'bot', 'support'], ['cart', 'bag', 'basket', 'checkout'], ['profile', 'account', 'user', 'settings', 'dashboard'],
            ['coupon', 'promotion', 'discount'], ['wp-login', 'login', 'log-in', 'signin', 'sign-in', 'signout', 'sign-out', 'logout', 'log-out'],
            ['contact', 'support', 'help'], ['search', 'find', 'query'], ['href', 'link'],
            ['upload', 'import', 'attach'], ['download', 'export'], ['wishlist', 'favorites', 'bookmarks'],
            ['compare', 'contrast'], ['review', 'rate', 'comment'], ['gallery', 'images', 'photos', 'portfolio'],
            ['newsletter', 'subscribe', 'unsubscribe'], ['post', 'submit', 'publish'], ['edit', 'update', 'modify'],
            ['delete', 'remove', 'discard'], ['share', 'refer', 'invite'], ['payment', 'checkout', 'purchase'],
            ['order', 'history', 'transactions'], ['tracking', 'shipment', 'delivery'], ['address', 'location', 'coordinates'],
            ['calendar', 'events', 'appointments'], ['booking', 'reservation', 'schedule'], ['forum', 'discussion', 'board'],
            ['report', 'feedback', 'complaint'], ['question', 'inquiry', 'support-ticket'], ['filter', 'sort', 'organize'],
            ['category', 'collection', 'department'], ['product', 'item', 'listing'], ['invoice', 'receipt', 'bill']]

keywordsPerFeature = {
    'registerForm': keywords[0],
    'rstPss': keywords[1],
    'chatBot': keywords[2],
    'bagCart': keywords[3],
    'accountSettings': keywords[4],
    'couponPromotion': keywords[5],
    'LoginLogout': keywords[6],
    'contactForm': keywords[7],
    'searchFeature': keywords[8],
    'linksPage': keywords[9],
    'fileUpload': keywords[10],
    'fileDownload': keywords[11],
    'wishlistFeature': keywords[12],
    'compareProducts': keywords[13],
    'reviewFeature': keywords[14],
    'imageGallery': keywords[15],
    'newsletterSubscription': keywords[16],
    'postSubmission': keywords[17],
    'editFeature': keywords[18],
    'deleteFeature': keywords[19],
    'shareReferral': keywords[20],
    'paymentProcess': keywords[21],
    'orderHistory': keywords[22],
    'shipmentTracking': keywords[23],
    'addressManagement': keywords[24],
    'calendarEvents': keywords[25],
    'bookingFeature': keywords[26],
    'forumDiscussion': keywords[27],
    'reportFeedback': keywords[28],
    'ticketSubmission': keywords[29],
    'filterSort': keywords[30],
    'categoryCollection': keywords[31],
    'productListing': keywords[32],
    'invoiceFeature': keywords[33],
}

# preprocess the test by normalizing it.
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\n\.]", " ", text)
    sentences = text.split("\n")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    sentences = [re.sub(r"^\s+|\s+$", "", sentence) for sentence in sentences]
    sentences = [s for s in sentences if len(s.split()) >= 3]
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


def label_threats(sentences, keywords, keywords_per_feature):
    df = pd.DataFrame(columns=["sentence"] + list(keywords_per_feature.keys()))
    
    for sentence in sentences:
        threat_labels = {}
        
        # Check each keyword category
        for category, category_keywords in keywords_per_feature.items():
            # Check if any keyword in the category is present in the sentence
            if any(keyword in sentence for keyword in category_keywords):
                # Label it as a positive match (1)
                threat_labels[category] = 1
            else:
                # Label it as a negative match (0)
                threat_labels[category] = 0
        
        # Create a new row in the DataFrame with the sentence and threat labels
        new_row = pd.DataFrame({"sentence": [sentence], **threat_labels})
        df = pd.concat([df, new_row], ignore_index=True)
    
    return df


def count_threats(df, keywords_per_feature):
    num_threats = {}
    total_threats = 0
    
    # Count the number of sentences classified as threats for each category
    for category in keywords_per_feature.keys():
        if category in df.columns:
            num_threats[category] = df[df[category] == 1].shape[0]
            total_threats += num_threats[category]
        else:
            num_threats[category] = 0
    
    # Print the counts for each category
    for category, threats_count in num_threats.items():
        if category in df.columns:
            print(f"Category: {category}")
            print(f"Number of threats: {threats_count}")
            print()
    
    # Print the total counts
    total_non_threats = df.shape[0] - total_threats
    total_sentences = df.shape[0]
    print("Total Counts:")
    print(f"Total number of threats: {total_threats}")
    print(f"Total number of non-threats: {total_non_threats}")
    print(f"Total number of sentences: {total_sentences}")


# Clean the dataframe by removing any noise left after initial prprocessing phase
def clean_dataframe(df):
    # Remove periods from sentences
    df["sentence"] = df["sentence"].apply(lambda x: x.replace(".", ""))

    # Remove sentences containing less than 3 words
    df = df[df["sentence"].str.split().apply(len) >= 3]

    # Remove duplicate sentences
    df = df.drop_duplicates(subset=["sentence"])

    # Reset the index
    df = df.reset_index(drop=True)

    return df


def remove_categories(df, min_threats):
    # Iterate over each column except the "sentence" column
    for column in df.columns:
        if column != "sentence":
            # Count the number of 1s in the column
            num_threats = df[column].sum()
            
            # Check if the number of threats is less than min_threats
            if num_threats < min_threats:
                # Remove the column from the dataframe
                df = df.drop(column, axis=1)
    
    return df


train_paths = [
    "Datasets/Documentation/CyberArk-and-Workfusion-IA2017Sunbird-integration-v3.pdf",
    "Datasets/Documentation/Desktop Analytics Solution - APA 7.0.pdf",
    "Datasets/Documentation/HPE_AWB_Guide_17.20.pdf",
    "Datasets/Documentation/SIS_1131_Deployment.pdf",
    "Datasets/Documentation/Whatfix Implementation and Security Document.pdf",
    "Datasets/Documentation/BSM_Integrations_HPOM.pdf",
    "Datasets/Documentation/fortiweb-v6.1.0-admin-guide.pdf",
    "Datasets/Documentation/db2z_12_adminbook.pdf",
    "Datasets/Documentation/pingfederate-110.pdf",
    "Datasets/Documentation/sg246098.pdf"
]

processed_sentences = []
for path in train_paths:
    processed_text = extract_text(path)
    processed_sentences.extend(processed_text)

labeled_df = label_threats(processed_sentences, keywords, keywordsPerFeature)
labeled_df = clean_dataframe(labeled_df)
labeled_df = remove_categories(labeled_df, 100)


count_threats(labeled_df, keywordsPerFeature)

labeled_df.to_csv("data.csv", index=False)
