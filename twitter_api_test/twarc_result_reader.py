import os
import json

from twarc import Twarc2, expansions

def page_reader(path):
    tweet_with = 0
    tweet_without = 0
    with open(path, 'r') as f:
        count = 0
        while True:
            print("tweet", count)
            tweet_string = f.readline()
            if not tweet_string:
                break
            tweet = json.loads(tweet_string)
            tweet_text = tweet["text"]

            if 'AAPL' in tweet_text or 'MSFT' in tweet_text:
                tweet_with += 1
            if 'AAPL' not in tweet_text or 'MSFT' not in tweet_text:
                tweet_without += 1
            count += 1
    return tweet_with, tweet_without

def query_result_loader(result_directory_path):
    # filenames without hidden macOS files starting with "."
    file_names = [file for file in os.listdir(result_directory_path) if file[0] != "."]
    total_with = 0
    total_without = 0
    for file_name in file_names:
        print(file_name)
        with_aapl, without_aapl = page_reader(os.path.join(result_directory_path, file_name))
        total_with += with_aapl
        total_without += without_aapl
    print(total_with, total_without)

if __name__ == "__main__":
    PAGE_0_PATH = './test_output/page_0'
    RESULT_PATH = './test_output'
     
    # page_reader(PAGE_0_PATH)
    query_result_loader(RESULT_PATH)