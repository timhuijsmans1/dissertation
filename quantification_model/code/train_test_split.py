from cgi import test
import json
from datetime import datetime
from posixpath import split

import pandas as pd

def df_builder(labelled_path):
    tweets = []
    with open(labelled_path, 'r') as f:
        for line in f:
            tweet_data = json.loads(line)
            label = tweet_data['label']
            datetime_string = tweet_data['created_at']
            date_string = datetime_string.split('T')[0]
            date_object = datetime.strptime(date_string, '%Y-%m-%d')
            original_text = tweet_data['original_text']
            tokens = tweet_data['processed_tokens']
            emoticon_list = tweet_data['emoticon_list']
            tweets.append((label, date_object, datetime_string, original_text, tokens, emoticon_list))w
    labelled_tweet_df = pd.DataFrame(tweets, columns=['label', 'date', 'datetime string', 'original text', 'tokens', 'emoticon list'])
    
    return labelled_tweet_df

def split_train_test(labelled_tweet_df):
    labelled_tweet_df.sort_values(by=['date'])
    end_row = int(len(labelled_tweet_df) * 0.8)

    train_df = labelled_tweet_df.iloc[0:end_row, :]
    test_df = labelled_tweet_df.iloc[end_row:, :]
    
    return train_df, test_df

def write_df_to_labelled(df, path):
    with open(path, 'w') as f:
        for index, row in df.iterrows():
            print(index)
            dict_to_write = {}
            dict_to_write['label'] = row['label']
            dict_to_write['original_text'] = row['original text']
            dict_to_write['created_at'] = row['datetime string']
            dict_to_write['processed_tokens'] = row['tokens']
            dict_to_write['emoticon_list'] = row['emoticon list']
            dict_as_string = json.dumps(dict_to_write) + "\n"
            f.write(dict_as_string)
    

if __name__ == "__main__":
    LABELLED_PATH = '../data/pre_processed_data/labelled_data.txt'
    TRAIN_OUTPUT_PATH = '../data/pre_processed_data/labelled_train_data.txt'
    TEST_OUTPUT_PATH = '../data/pre_processed_data/labelled_test_data.txt'
    labelled_tweet_df = df_builder(LABELLED_PATH)
    train_df, test_df = split_train_test(labelled_tweet_df)
    write_df_to_labelled(train_df, TRAIN_OUTPUT_PATH)
    write_df_to_labelled(test_df, TEST_OUTPUT_PATH)