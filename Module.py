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
                if filepath.endswith(".pdf"):
                    tokens = self.dp.process_pdf(filepath)


m = Module("Datasets/Documentation")
m.preprocess_data()