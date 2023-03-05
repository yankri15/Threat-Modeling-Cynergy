import os
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from PyPDF2 import PdfFileReader



class SpacyEngine:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text):
        # Perform basic text cleaning
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize_text(self, text):
        # Tokenize the text using spaCy's built-in tokenization method
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        return tokens

    def remove_stopwords(self, tokens):
        # Remove stop words from the tokens
        return [token for token in tokens if token not in STOP_WORDS]

    def lemmatize_tokens(self, tokens):
        # Lemmatize the tokens using spaCy's built-in lemmatization method
        doc = self.nlp(" ".join(tokens))
        lemmas = [token.lemma_ for token in doc]
        return lemmas

    def process_text(self, text):
        # Process the text using spaCy's nlp 
        doc = self.nlp(text)

        # Iterate over the entities in the text
        for ent in doc.ents:
            # Print the entity text and label
            print(ent.text, ent.label_)

        # Add any additional processing or analysis here later

    def process_pdf(self, path):
        with open(path, "rb") as f:
            reader = PdfFileReader(f)
            for page in reader.pages:
                text = page.text
                text = self.preprocess_text(text)
                tokens = self.tokenize_text(text)
                tokens = self.remove_stopwords(tokens)
                tokens = self.lemmatize_tokens(tokens)
                # Do whatever you want with the processed tokens here



# Old code

# class SpacyEngine:
#     def __init__(self):
#         self.nlp = spacy.load("en_core_web_sm")

#     def preprocess_text(self, text):
#         # Perform basic text cleaning
#         text = text.lower()
#         text = re.sub(r"[^a-zA-Z0-9]", " ", text)
#         text = re.sub(r"\s+", " ", text)
#         return text

#     def remove_stopwords(self, tokens):
#         # Remove stop words from a list of tokens.
#         doc = self.nlp(" ".join(tokens))
#         return [token.text for token in doc if not token.is_stop]

#     def tokenize_text(self, text):
#         # Tokenize the text using spaCy's built-in tokenization method
#         doc = self.nlp(text)
#         tokens = [token.text for token in doc]
#         return tokens

#     def process_text(self, text):
#         # Process the text using spaCy's nlp 
#         doc = self.nlp(text)

#         # Iterate over the entities in the text
#         for ent in doc.ents:
#             # Print the entity text and label
#             print(ent.text, ent.label_)

#         # Add any additional processing or analysis here later

#     def process_dataset(self, path):
#         for subdir, dirs, files in os.walk(path):
#             for file in files:
#                 filepath = subdir + os.sep + file
#                 if filepath.endswith(".pdf"):
#                     with open(filepath, "rb") as f:
#                         pdf = PyPDF2.PdfFileReader(f)
#                         text = ""
#                         for page in pdf.pages:
#                             text += page.extract_text()
#                     text = self.preprocess_text(text)
#                     self.process_text(text)

'''
class PytorchBERT():
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dataset = None
        
    def load_dataset(self, dataset_path):
        self.dataset = []
        for foldername in os.listdir(dataset_path):
            foldername = os.path.join(dataset_path, foldername)
            if os.path.isdir(foldername):
                for filename in os.listdir(foldername):
                    filename = os.path.join(foldername, filename)
                    with open(filename, 'r', encoding="utf-8", errors='ignore') as f:
                        text = f.read()
                        text = self.preprocess_text(text)
                        self.dataset.append(text)

                        
    def process_text(self):
        for text in self.dataset:
            input_ids = self.tokenizer.encode(text, add_special_tokens=True)
            input_ids_tensor = torch.tensor([input_ids])
            logits = self.model(input_ids_tensor)[0]
            print(logits)


class GensimEngine:
    def __init__(self):
        self.model = None
        self.dataset = None
    
    def load_language_model(self, model_path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    
    def load_dataset(self, dataset_path):
        with open(dataset_path, 'r') as f:
            self.dataset = f.readlines()
    
    def preprocess_text(self, text):
        # Perform basic text cleaning
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text
    
    def tokenize_text(self, text):
        # Tokenize the text using Gensim's built-in tokenization method
        tokens = gensim.utils.simple_preprocess(text)
        return tokens
    
    def process_text(self, text):
        # Process the text using the Gensim model
        tokens = self.tokenize_text(text)
        vectors = [self.model[token] for token in tokens if token in self.model.vocab]
        return vectors
    
    def process_dataset(self):
        dataset_vectors = []
        for text in self.dataset:
            vectors = self.process_text(text)
            dataset_vectors.append(vectors)
            return dataset_vectors

    def test_performance(self):
        start_time = time.time()
        dataset_vectors = self.process_dataset()
        end_time = time.time()
        total_time = end_time - start_time
        print("Processing time: {:.2f} seconds".format(total_time))
        return total_time


def test_spacy():
    """
    Testing the spacy engine
    """

    processor = SpacyEngine()
    dataset_path = "Datasets/20_newsgroups"
    processor.process_dataset(dataset_path)


def test_pytorchBERT():
    """
    Testing the PytorchBERT engine
    """
    bert = PytorchBERT()
    bert.load_dataset('Datasets/20_newsgroups')
    bert.process_text()


def test_gensim_engine():
    """
    Testing the GensimEngine 
    """
    # Initialize the GensimEngine object
    gensim_engine = GensimEngine()
    
    # Load the pre-trained language model
    model_path = 'path/to/word2vec_model.bin'
    gensim_engine.load_language_model(model_path)
    
    # Load the dataset
    dataset_path = 'Datasets/20_newsgroups'
    gensim_engine.load_dataset(dataset_path)
    
    # Test the performance of the engine on the dataset
    processing_time = gensim_engine.test_performance()
    print("GensimEngine processed the dataset in {:.2f} seconds.".format(processing_time))
'''