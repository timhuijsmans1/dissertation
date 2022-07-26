import json
from pyexpat import model
import numpy as np

from scipy import stats


def read_scores(path):
    with open(path, 'r') as f:
        scores = json.load(f)
    positive_scores = np.array([tweet_data['score'] for tweet_data in scores['pos']])
    negative_scores = np.array([tweet_data['score'] for tweet_data in scores['neg']])
    return positive_scores, negative_scores

def calculate_statistics(score_array):
    avg = np.mean(score_array)
    std_dev = np.std(score_array)
    min_score = np.min(score_array)
    max_score = np.max(score_array)
    mode = stats.mode(score_array).mode[0]
    return {'avg': avg, 'std_dev': std_dev, 'min': min_score, 'max': max_score, 'mode': mode}

if __name__ == "__main__":
    LABEL_COUNTS_PATH = '../data/labelled_data/scores/union_emoticon_stats.json'
    pos_arr, neg_arr = read_scores(LABEL_COUNTS_PATH)
    print('positive stats:')
    print(calculate_statistics(pos_arr))
    print('negative stats:')
    print(calculate_statistics(neg_arr))
    