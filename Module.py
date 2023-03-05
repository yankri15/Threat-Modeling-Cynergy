import os
from DataPreProcessing import SpacyEngine


class Module:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dp = SpacyEngine()

    def preprocess_data(self):
        for subdir, dirs, files in os.walk(self.data_path):
            for file in files:
                filepath = subdir + os.sep + file
                with open(filepath, "r", encoding="ISO-8859-1") as f:
                    text = f.read()
                    text = self.dp.preprocess_text(text)
                    tokens = self.dp.tokenize_text(text)
                    tokens = self.dp.remove_stopwords(tokens)
                    tokens = self.dp.lemmatize_tokens(tokens)
                    # Do whatever you want with the processed tokens here

    def process_data(self):
        for subdir, dirs, files in os.walk(self.data_path):
            for file in files:
                filepath = subdir + os.sep + file
                with open(filepath, "r", encoding="ISO-8859-1") as f:
                    text = f.read()
                    self.dp.process_text(text)
                    
    def process_pdf(self, path):
        self.dp.process_pdf(path)


module = Module('Datasets/Documentation')
module.preprocess_data()
module.process_data()
module.process_pdf('Datasets/Documentation')