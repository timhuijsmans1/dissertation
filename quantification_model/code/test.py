import os
import json
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def test_n_grams(engineered_path, train_path, labelled_path):
    with open(engineered_path, 'r') as f_eng, open(train_path, 'r') as f_train, open(labelled_path, 'r') as f_labelled: 
        count = 0
        while count < 99:
            labelled_tokens = json.loads(f_labelled.readline())["processed_tokens"]
            engineered_line = json.loads(f_eng.readline())
            engineered_tokens = engineered_line["n-grams"]
            engineered_sum = sum(engineered_line['engineered features'].values())
            feature_list_freqs = [int(float(feature.split(":")[1])) for feature in f_train.readline().split(" ")[1:]]
            number_of_tokens = sum(feature_list_freqs)
            print('labelled: ', (len(labelled_tokens) - 1) + len(labelled_tokens))
            print('engineered: ', len(engineered_tokens))
            print('features: ', number_of_tokens - engineered_sum)
            print("-" * 20)

def data_splitting(labelled_path):
    tweets = []
    with open(labelled_path, 'r') as f:
        for line in f:
            tweet_data = json.loads(line)
            label = tweet_data['label']
            date_string = tweet_data['created_at'].split('T')[0]
            date_object = datetime.strptime(date_string, '%Y-%m-%d')
            
            tweets.append((label, date_object))
            print(len(tweets))
    label_df = pd.DataFrame(tweets, columns=['label', 'date'])

    return label_df

def plot_labels(label_df):
    plt.figure()
    sns.histplot(data=label_df, x='date', hue='label', bins=50, palette='bright')
    plt.show()        

def count_lines(path):
    with open(path, 'r') as f:
        count = 0
        for line in f:
            count += 1
        print(count)
if __name__ == "__main__":
    ENGINEERED_PATH = "../data/pre_processed_data/engineered.txt"
    LABELLED_PATH = "../data/pre_processed_data/labelled_data.txt"
    TRAIN_OUTPUT_PATH = "../data/train_test_data/train.dat"

    # test_n_grams(ENGINEERED_PATH, TRAIN_PATH, LABELLED_PATH)
    # df = data_splitting(ENGINEERED_PATH)
    # plot_labels(df)
    count_lines(TRAIN_OUTPUT_PATH)