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
                print(tweet_data['created_at'])

                if label == -1:
                    neg_count += 1
                    
                else:
                    pos_count += 1
                    print(tweet_data['original_text'])
                    print("-" * 40)

        return

if __name__ == "__main__":
    TRAINING_DATA_PATH = '../data/labelled_data/labelled_data_union.txt'
    explorer = dataExplorer()
    explorer.data_reader(TRAINING_DATA_PATH)