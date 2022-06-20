import os
import json

class dataExplorer:

    def data_reader(self, training_data_path):
        with open(training_data_path, 'r') as f:
            while True:
                labelled_tweet = f.readline()
                if not labelled_tweet:
                    break
                else: 
                    label = labelled_tweet[0]
                    token_string = json.loads(labelled_tweet[2:])
                
                if label == '-':
                    print(label, token_string)
                    print('-' * 50)

if __name__ == "__main__":
    TRAINING_DATA_PATH = '../data/labelled_data/labelled_data.txt'
    explorer = dataExplorer()
    explorer.data_reader(TRAINING_DATA_PATH)