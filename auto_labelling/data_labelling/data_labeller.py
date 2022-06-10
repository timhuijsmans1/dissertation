import json
import os

class dataLabeller:

    def __init__(self, data_path):
        self.file_reader(data_path)    
    
    def file_reader(self, path):
        with open(path, 'r') as f:
            count = 1
            while True:
                line = f.readline()
                if not line:
                    break
                else:
                    tweet_data = json.loads(line)
                    if count < 100:
                        print(tweet_data['text'])
                        print("\n -------------------------- \n")
                    count += 1

if __name__ == "__main__":
    DATA_PATH = "../data_collector/collected_data/filtered_search_results.txt"
    dataLabeller(DATA_PATH)