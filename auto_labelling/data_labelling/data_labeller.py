import json
import os
import re

class dataLabeller:

    def __init__(
            self, 
            data_path, 
            lexicon_path, 
            output_path, 
            threshold, 
            pos_emoticon, 
            neg_emoticon
        ):
        if os.path.exists(output_path):
            os.remove(output_path) # removes the existing labelled file to allow
                                   # for append mode in writing.

        self.lexicon = self.lexicon_reader(lexicon_path)
        self.lexicon_score_threshold = threshold
        self.output_path = output_path
        self.pos_emoticon = pos_emoticon
        self.neg_emoticon = neg_emoticon
        self.file_reader(data_path)

    def lexicon_reader(self, lexicon_path):
        lexicon = {}
        with open(lexicon_path, 'r') as f:
            header = f.readline()

            while True:
                line = f.readline()
                if not line:
                    break
                else:
                    word, score = line.split(",")[1:]
                    word = word.strip("\"")
                    score = float(score.strip("\n"))
                    lexicon[word] = score
        
        return lexicon

    def emoticon_extracter(self, tweet_string):
        # regex to find any emoticon in the tweet provided
        expression = r"\\ud[\d\w]{3}\\ud[\d\w]{3}"
        emoticon_list = re.findall(expression, tweet_string)
        # recompiles the emoticons as browser representation
        emoticon_list = [json.loads("\"" + emoticon + "\"") for emoticon in emoticon_list]
    
        return emoticon_list

    def lexicon_labeller(self, tweet_text):
        # calculate the total tweet sentiment score
        tweet_score = 0
        for token in tweet_text:
            token_score = self.lexicon.get(token, 0)
            tweet_score += token_score

        # set sentiment labels based of score
        if tweet_score > 0:
            label = 1
        elif tweet_score < 0:
            label = -1
        else:
            label = 0
        
        return label, tweet_score
    
    def emoticon_labeller(self, emoticon_list):
        emoticon_score = 0
        used_emoticons = [] # keeps track of the emoticons used for score calculation
        for emoticon in emoticon_list:
            if emoticon in self.pos_emoticon:
                used_emoticons.append(emoticon)
                emoticon_score += 1
            if emoticon in self.neg_emoticon:
                used_emoticons.append(emoticon)
                emoticon_score -= 1
        if emoticon_score > 0:
            label = 1
        if emoticon_score < 0:
            label = -1
        if emoticon_score == 0:
            label = None

        return label, emoticon_score, used_emoticons

    def label_writer(
        self, 
        label, 
        output_path, 
        tweet_text, 
        username, 
        emoticon_score, 
        used_emoticons
        ):
        with open(output_path, 'a+') as f:
            tuple_to_write = (
                label, 
                username, 
                tweet_text, 
                used_emoticons, 
                emoticon_score
            )
            string_of_tuple = json.dumps(tuple_to_write) + "\n"
            f.write(string_of_tuple)

        return
        
    def file_reader(self, path):
        lexicon_containing_tweets = 0
        class_balance_counter = {-1: 0, 0: 0, 1: 0}

        with open(path, 'r') as f:
            count = 1
            while True:
                line = f.readline()
                # break the while loop at the end of the file
                if not line:
                    break
                else:
                    # splits all tokens in the tweets
                    tweet_data = json.loads(line)
                    tweet_text = tweet_data['text']
                    username = tweet_data['author']['username']
                    tokens = tweet_text.split(" ")

                    # extract the emoticons from the tweet_text 
                    tweet_text_encoded = json.dumps(tweet_text)
                    emoticon_list = self.emoticon_extracter(tweet_text_encoded)

                    # labels and writes tweet, keeps track of class balance
                    lexicon_label, lexicon_score = self.lexicon_labeller(tokens)
                    emoticon_label, emoticon_score, used_emoticons = (
                        self.emoticon_labeller(emoticon_list)
                    )
                    
                    if lexicon_label == emoticon_label:
                        self.label_writer(
                            lexicon_label,
                            self.output_path, 
                            tokens, 
                            username, 
                            emoticon_score, 
                            used_emoticons
                        )
                        class_balance_counter[lexicon_label] += 1 
                    count += 1
                    print(count)
        print(class_balance_counter)

        return

if __name__ == "__main__":
    # global variables
    DATA_PATH = (
        "../data/collected_data/filtered_search_results_06-19-2022_20;02;51.txt"
    )
    OUTPUT_PATH = "../data/labelled_data/labelled_data.txt"
    FINANCIAL_LEXICON_PATH = "../data/fin_sent_lexicon/lexicons/lexiconWNPMINW.csv"
    LEXICON_SCORE_THRESHOLD = 0.01
    POS_EMOTICON_LIST = ["😀", "😃", "😄", "😁", "🙂"]
    NEG_EMOTICON_LIST = ["😡", "😤", "😟", "😰", "😨", "😖", "😩", "🤬", "😠", "💀", "👎", "😱"]

    # exucute data labelling
    dataLabeller(
        DATA_PATH, 
        FINANCIAL_LEXICON_PATH, 
        OUTPUT_PATH, 
        LEXICON_SCORE_THRESHOLD, 
        POS_EMOTICON_LIST, 
        NEG_EMOTICON_LIST
    )