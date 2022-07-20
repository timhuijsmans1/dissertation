import os
import json

class dataExplorer:

    def data_reader(self, training_data_path):
        neg_count = 0
        pos_count = 0
        count = 0
        with open(training_data_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                tweet_data = json.loads(line)
                label = tweet_data['label']

                if label == -1:
                    neg_count += 1
                else:
                    pos_count += 1

                count += 1
                print(count)
        print(neg_count)
        print(pos_count)
        return
                

if __name__ == "__main__":
    TRAINING_DATA_PATH = '../../quantification_model/data/pre_processed_data/labelled_data.txt'
    explorer = dataExplorer()
    explorer.data_reader(TRAINING_DATA_PATH)