import subprocess
import os
import time

# path to the scripts
preprocessing_script_path = 'DataPreProcessing.py'
model_script_path = 'DnnModel.py'

# path to the status file from preprocessing
status_file_path = 'preprocessing_done.txt'

# delete the status file if it exists
try:
    os.remove(status_file_path)
except FileNotFoundError:
    pass

# run the data preprocessing script
subprocess.run(['python', preprocessing_script_path], check=True)

# wait for the status file to be created by the preprocessing script
while not os.path.exists(status_file_path):
    time.sleep(1)

# run the model script
subprocess.run(['python', model_script_path], check=True)
