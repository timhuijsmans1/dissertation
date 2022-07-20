import json

def build_vocabulary(tweet_path):
    vocabulary = set()
    with open(tweet_path, 'r') as f:
        for line in f:
            tokens = set(json.loads(line)['processed_tokens'])
            print(tokens)
            vocabulary |= tokens
    return vocabulary

def write_voc(vocabulary):
    with open("../data/pre_processed_data/vocabulary.txt", 'w') as f:
        json.dump(list(vocabulary), f)

if __name__ == "__main__":
    TWEET_PATH = '../data/pre_processed_data/labelled_data.txt'
    voc = build_vocabulary(TWEET_PATH)
    write_voc(voc)