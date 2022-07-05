import json
import os
import re
import emoji as emoji_lib
import regex

class dataLabeller:

    def __init__(
            self, 
            data_path, 
            lexicon_path, 
            output_path,
            negation_file_path,
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
        self.negation_indicators = self.negation_indicator_loader(negation_file_path)
        self.file_reader(data_path)
    

    def negation_indicator_loader(self, negation_file_path):
        negation_indicators = set()
        with open(negation_file_path, 'r') as f:
            for line in f:
                negation_indicators.add(line.strip('\n '))
        return negation_indicators

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

    def emoji_extractor(self, tweet_text):
        emoji_list = []
        data = regex.findall(r'\X', tweet_text)
        for word in data:
            if any(char in emoji_lib.UNICODE_EMOJI['en'] for char in word):
                emoji_list.append(word)
        flags = regex.findall(u'[\U0001F1E6-\U0001F1FF]', tweet_text) 
        
        emoji_list = emoji_list + flags

        for emoji in emoji_list:
            tweet_text = tweet_text.replace(emoji, ' ') # replace with a space 
                                                        # to make sure that potential
                                                        # words around the emoji 
                                                        # separated
    
        return tweet_text, emoji_list

    def cashtag_extractor(self, tweet_text):
        cashtag_list = []
        cashtag_list = regex.findall(r'\$[a-zA-Z0-9]+', tweet_text)

        for cashtag in cashtag_list:
            tweet_text = tweet_text.replace(cashtag, '')

        return tweet_text, cashtag_list

    def string_processing(self, tweet_text):
        # replace all linebreaks and tabs in tweet by spaces
        tweet_text = tweet_text.replace('\n', ' ')
        # replace all tabs in tweet by spaces
        tweet_text = tweet_text.replace('\t', ' ')
        # remove RT tag from tweet
        tweet_text = tweet_text.replace('RT', '')
        # replace all multiple spaces in tweet by single spaces
        tweet_text = re.sub(r'\s+', ' ', tweet_text)
        # remove all web links from tweet
        tweet_text = re.sub(r'http\S+', '', tweet_text)
        # remove all usernames from tweet  
        tweet_text = re.sub(r'@\S+', '', tweet_text)
        # replace all underscores by a space
        tweet_text = tweet_text.replace('_', ' ')

        return tweet_text

    def tokenisation(self, tweet_text):

        # split tweet on spaces
        tweet_tokens = tweet_text.split(' ')

        # strip all punctuation of tokens
        tweet_tokens = [token.strip(',.!;:?()# ') for token in tweet_tokens]

        # remove all tokens not containing any alphanumeric characters
        tweet_tokens = [token.lower() for token in 
                        tweet_tokens if re.search(r'[a-zA-Z0-9]', token)]

        # remove all empty tokens
        tweet_tokens = [token for token in tweet_tokens if token != '']

        return tweet_tokens

    def negator(self, token):
        return "neg_" + token

    def negation_handling(self, tokens):
        negated_tokens = []
        
        i = 0
        while i < len(tokens):
            if tokens[i] in self.negation_indicators:
                
                # antepenultimate token is negation indicator
                if i + 3 == len(tokens):
                    negated_tokens.append(self.negator(tokens[i + 1]))
                    negated_tokens.append(self.negator(tokens[i + 2]))
                    break

                # penultimate token is negation indicator
                elif i + 2 == len(tokens):
                    negated_tokens.append(self.negator(tokens[i + 1]))
                    break

                # last token is negation indicator
                elif i + 1 == len(tokens):
                    break
                    
                else:
                    negated_tokens.append(self.negator(tokens[i + 1]))
                    negated_tokens.append(self.negator(tokens[i + 2]))
                    i += 3 # skip next two tokens

            else:
                negated_tokens.append(tokens[i])
                i += 1

        return negated_tokens

    def num_handling(self, tokens):
        pattern_pos = r'^\+[1-9]\d*(\.\d+)?'
        pattern_neg = r'^\-[1-9]\d*(\.\d+)?'
        i = 0
        while i < len(tokens):
            if re.search(pattern_pos, tokens[i]):
                print('original: ', tokens)
                tokens[i] = 'posnum'
            elif re.search(pattern_neg, tokens[i]):
                print('original: ', tokens)
                tokens[i] = 'negnum'
            i += 1
        return tokens

    def tweet_pre_processing(self, tweet_text):

        tweet_text = self.string_processing(tweet_text)

        # extract and remove emojis and cashtags from the tweet text
        tweet_text, emoticon_list = self.emoji_extractor(tweet_text)
        tweet_text, cashtag_list = self.cashtag_extractor(tweet_text)

        tweet_tokens = self.tokenisation(tweet_text)
        tweet_tokens = self.negation_handling(tweet_tokens)
        tweet_tokens = self.num_handling(tweet_tokens)

        tweet_data = {
                'tweet_tokens': tweet_tokens, 
                'emoticon_list': emoticon_list,
                'cashtag_list': cashtag_list
        }

        return tweet_data

    def label_writer(
        self, 
        output_path, 
        dict_to_write
        ):
        string_of_dict = json.dumps(dict_to_write) + "\n"
        with open(output_path, 'a+') as f:
            f.write(string_of_dict)

        return
        
    def file_reader(self, path):
        class_balance_counter = {-1: 0, 0: 0, 1: 0}

        with open(path, 'r') as f:
            count = 1
            while True:
                output_to_write = {}
                line = f.readline()
                # break the while loop at the end of the file
                if not line:
                    break
                else:
                    tweet_data = json.loads(line)
                    tweet_text = tweet_data['text']
                    tweet_creation_date = tweet_data['created_at']
                    pre_processed_data = self.tweet_pre_processing(tweet_text)
                    tweet_tokens = pre_processed_data['tweet_tokens']
                    emoticon_list = pre_processed_data['emoticon_list']
                    cashtag_list = pre_processed_data['cashtag_list']
                    
                    # label the tweet
                    lexicon_label, lexicon_score = self.lexicon_labeller(tweet_tokens)
                    emoticon_label, emoticon_score, used_emoticons = self.emoticon_labeller(emoticon_list)

                    if lexicon_label == emoticon_label:
                        label = lexicon_label
                        dict_to_write = {
                            'label': label, 
                            'original_text': tweet_text,
                            'created_at': tweet_creation_date,
                            'processed_tokens': tweet_tokens,
                            'emoticon_list': emoticon_list,
                            'cashtag_list': cashtag_list
                        }
                        self.label_writer(self.output_path, dict_to_write)

                        if label == 1:
                            class_balance_counter[1] += 1
                        elif label == -1:
                            class_balance_counter[-1] += 1
                        elif label == 0:
                            class_balance_counter[0] += 1

                    count += 1
        print(class_balance_counter)

        return

if __name__ == "__main__":
    # global variables
    DATA_PATH = (
        "../data/preprocessed_data/final_data_06-26-2022_15;11;01.txt"
    )
    OUTPUT_PATH = "../data/labelled_data/labelled_data.txt"
    FINANCIAL_LEXICON_PATH = "../data/fin_sent_lexicon/lexicons/lexiconWNPMINW.csv"
    NEG_INDICATOR_PATH = "./negation_ind.txt"
    LEXICON_SCORE_THRESHOLD = 0.01
    POS_EMOTICON_LIST = ["😀", "😃", "😄", "😁", "🙂"]
    NEG_EMOTICON_LIST = ["😡", "😤", "😟", "😰", "😨", "😖", "😩", "🤬", "😠", "💀", "👎", "😱"]

    # exucute data labelling
    dataLabeller(
        DATA_PATH,
        FINANCIAL_LEXICON_PATH, 
        OUTPUT_PATH, 
        NEG_INDICATOR_PATH,
        LEXICON_SCORE_THRESHOLD, 
        POS_EMOTICON_LIST, 
        NEG_EMOTICON_LIST
    )