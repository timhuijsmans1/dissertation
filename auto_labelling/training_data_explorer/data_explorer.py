import os
import json

class dataExplorer:

    def data_reader(self, training_data_path):
        with open(training_data_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                tweet_data = json.loads(line)
                label = tweet_data[0]

                if label == -1:
                    print(tweet_data)
                    print('-' * 30)
        
        return
                

if __name__ == "__main__":
    TRAINING_DATA_PATH = '../data/labelled_data/labelled_data.txt'
    explorer = dataExplorer()
    explorer.data_reader(TRAINING_DATA_PATH)