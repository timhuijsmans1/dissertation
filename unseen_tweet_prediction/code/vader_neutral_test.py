import json
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def combine_tokens(data_path):
    tweets = []
    with open(data_path, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            text = ' '.join(tweet['processed_tokens'])
            tweet['processed_string'] = text
            tweets.append(tweet)
    return tweets

def predict_sentiments(tweets):
    analyzer = SentimentIntensityAnalyzer()
    for tweet in tweets:
        vs = analyzer.polarity_scores(tweet['processed_string'])
        print("-" * 30)
        print(vs)
        print(tweet['original_text'])

def repeated_cashtag_checker(tweets):
    pattern = re.compile(r"\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+")
    count = 0
    for tweet in tweets:
        if pattern.search(tweet['original_text']):
            print(tweet['original_text'])
            print("-" * 30)
            count += 1
    print(len(tweets), count)
if __name__ == "__main__":
    DATA_PATH = "../data/pipeline_output/AAPL/engineered_data/2021-07-21.txt"
    tweets = combine_tokens(DATA_PATH)
    # predict_sentiments(tweets)
    repeated_cashtag_checker(tweets)