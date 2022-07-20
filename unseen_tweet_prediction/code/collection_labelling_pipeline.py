import json
import os
import datetime
import regex
import re
import emoji as emoji_lib
import numpy as np

from twarc import Twarc2, expansions
from itertools import islice
from scipy import sparse
from nltk import bigrams

class TweetCollector():

    def __init__(self, 
                client, 
                start_datetime, 
                end_datetime,
                output_path,
        ):

        self.client = client
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.output_path = output_path

    def ticker_query_compiler(self, ticker):
        query = ticker + " -is:retweet lang:en"
        return query

    def execute_search(self, query):
        search_results = self.client.search_all(
                            query=query, 
                            start_time=self.start_datetime, 
                            end_time=self.end_datetime, 
                            max_results=100
        )

        return search_results

    def result_writer(self, query):
        with open(self.output_path, 'w+') as f: 
            search_results = self.execute_search(query)
            # this is required because of the result generator returned
            # by the Twitter API
            for page in search_results:
                result = expansions.flatten(page)

                # check if tweets are valid and write valid tweets
                for tweet in result:
                    f.write('%s\n' % json.dumps(tweet))

    def execute_company_search(self, ticker):
        query = self.ticker_query_compiler(ticker)
        self.result_writer(query)

class duplicatePreProcessor:

    def __init__(
        self, 
        raw_data_path, 
        output_path, 
        cosine_threshold,
        high_freq_threshold,
        ):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.cosine_threshold = cosine_threshold
        self.high_freq_threshold = high_freq_threshold

        self.unique_tweet_vectors = []
        self.vocabulary = set()

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
        tweet_tokens = [token.strip(',.!;:?()#& ') for token in tweet_tokens]

        # remove all tokens not containing any alphanumeric characters
        # and cast all tokens to lowercase tokens
        tweet_tokens = [token.lower() for token in 
                        tweet_tokens if re.search(r'[a-zA-Z0-9]', token)]

        # remove all empty tokens
        tweet_tokens = [token for token in tweet_tokens if token != '']         

        return tweet_tokens

    def tweet_pre_processing(self, tweet_text):

        tweet_text = self.string_processing(tweet_text)

        # extract and remove emojis and cashtags from the tweet text
        tweet_text, emoticon_list = self.emoji_extractor(tweet_text)
        tweet_text, cashtag_list = self.cashtag_extractor(tweet_text)

        tweet_tokens = self.tokenisation(tweet_text)
        tweet_data = {
                'tweet_tokens': tweet_tokens, 
                'emoticon_list': emoticon_list,
                'cashtag_list': cashtag_list
        }

        return tweet_data

    def tweet_dict_to_write(self, full_tweet_dict):
        # remove the redundant stuff of the line, and add the relevant stuff
        # such as the tokens etc.
        tweet_text = full_tweet_dict['text']
        tweet_creation_date = full_tweet_dict['created_at']
        pre_processing_data = full_tweet_dict['pre_processing_data']
        tweet_tokens = pre_processing_data['tweet_tokens']
        emoticon_list = pre_processing_data['emoticon_list']
        cashtag_list = pre_processing_data['cashtag_list']
        dict_to_write = dict_to_write = {
            'original_text': tweet_text,
            'created_at': tweet_creation_date,
            'processed_tokens': tweet_tokens,
            'emoticon_list': emoticon_list,
            'cashtag_list': cashtag_list
        }
        return dict_to_write

    def pre_process_collection(self):
        with open(self.raw_data_path, 'r') as f_in, open('./pre_processed.txt', 'w') as f_out:
            self.tweet_count = 0
            while True:
                line = f_in.readline()
                if not line:
                    break
                else:
                    tweet_data = json.loads(line)
                    tweet_text = tweet_data['text']

                    # run pre_processing on individual tweets
                    pre_processing_data = self.tweet_pre_processing(tweet_text)
                    pre_processed_tokens = pre_processing_data['tweet_tokens']
                    self.vocabulary |= set(pre_processed_tokens)
                    
                    # write tweet data to the pre_processed_data file
                    tweet_data['pre_processing_data'] = pre_processing_data
                    dict_to_write = self.tweet_dict_to_write(tweet_data)
                    f_out.write(json.dumps(dict_to_write) + '\n')

                    self.tweet_count += 1
                    
                    # print tweet count every 1000 tweets
                    if self.tweet_count % 1000 == 0:
                        print('Processed {} tweets'.format(self.tweet_count))

        return

    def set_word2index(self):
        self.word2index = {}
        for i, word in enumerate(self.vocabulary):
            self.word2index[word] = i

        # swap all keys and values of word2index
        self.index2word = {v: k for k, v in self.word2index.items()}
        
        return

    def dense_vector_from_tokens(self, tokens):
        dense_tweet_vector = np.zeros(len(self.vocabulary))
        for token in tokens:
            token_index = self.word2index[token]
            dense_tweet_vector[token_index] += 1
        return dense_tweet_vector

    def cosine_similarity_check(self, threshold, vector_1, vector_2):
        """
        calculate cosine similarity between vector 1 and 2: return
        True if higher than threshold and False if lower than threshold
        """
        cosine_similarity = np.dot(vector_1, vector_2) \
            / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))

        return cosine_similarity >= threshold
    
    def duplicate_check(self, sparse_vector_list, vector_to_check):
        # assume that the tweet is no duplicate until tested
        new_tweet = True
        for i, vector in enumerate(sparse_vector_list):
            
            # break the loop if the tweet vector is a duplicate
            dense_existing_vector = vector[0].todense()[0]

            if self.cosine_similarity_check(
                        self.cosine_threshold, 
                        dense_existing_vector,
                        vector_to_check
                ):
                new_tweet = False
                break

        return new_tweet, i

    def sparse_in_list(self, list, sparse_vector):
        for vector in list:
            if (vector.todense() == sparse_vector.todense()).all():
                return True
        return False

    def remove_numerical(self, tweet_tokens):
        # remove all tokens that contain any numbers,
        # to compare the core text and not numbers for similarity
        non_numerical_tokens = []
        pattern_pos = r'[1-9]\d*(\.\d+)?'
        for token in tweet_tokens:
            if not re.search(pattern_pos, token):
                non_numerical_tokens.append(token)
        return non_numerical_tokens

    def duplicate_filter(self):
        sparse_vectors = []
        high_freq_vectors = []
        with open('./pre_processed.txt', 'r') as f_in, open(self.output_path, 'w') as f_out:
            count = 0
            while True:
                # limit the size of the high freq vectors to avoid 
                # time issues
                if len(high_freq_vectors) > 200:
                    high_freq_vectors = []
                line = f_in.readline()
                if not line:
                    break
                else:
                    tweet_data = json.loads(line)
                    pre_processed_tokens = (
                        tweet_data['processed_tokens']
                    )
                    non_numerical_tokens = self.remove_numerical(pre_processed_tokens)
                    dense_new_vector = self.dense_vector_from_tokens(
                                                        non_numerical_tokens
                    )

                    # check if duplicate or similar, always add the first
                    # vector to the list of sparse vectors
                    if sparse_vectors:
                        new_tweet, duplicate_index = self.duplicate_check(
                                            sparse_vectors, 
                                            dense_new_vector,
                        )
                        # if the tweet vector is not similar, add it to the list
                        # and write the corresponding tweet data to the output file
                        if new_tweet:
                            sparse_vectors.append([sparse.coo_array(dense_new_vector), 1])
                            f_out.write(line)
                        # if the tweet vector is similar, increase the frequency of the
                        # vector in the list and 
                        else:
                            sparse_vectors[duplicate_index][1] += 1
                            if sparse_vectors[duplicate_index][1] > self.high_freq_threshold:
                                if not self.sparse_in_list(high_freq_vectors, sparse_vectors[duplicate_index][0]):
                                    high_freq_vectors.append(sparse_vectors[duplicate_index][0])
                    else:
                        sparse_vectors.append([sparse.coo_array(dense_new_vector), 1])
                        f_out.write(line)
                
                    count += 1
                    if count % 1000 == 0:
                        # this stores the history of high freq tweets in the 
                        # sparse vector list to use in the next 1000 duplicate 
                        # checks. the freq of one does not matter, as these tweets 
                        # will remain in the high freq tweets anyways.
                        sparse_vectors = [[vector, 1] for vector in high_freq_vectors]
                        print("tweets in most recent tweets : ", len(sparse_vectors))
                    if count % 100 == 0:    
                        print("tweets processed: ", count)
                        print("tweets in high freq tweets: ", len(high_freq_vectors))
                        print("tweets in most recent tweets : ", len(sparse_vectors))
         
        return

class featureEngineering:

    def __init__(self, tweet_path, engineered_output_path, negator_indicator_path):
        self.input_path = tweet_path
        self.output_path = engineered_output_path
        self.negation_indicators = self.negation_indicator_loader(negator_indicator_path)

    def negation_indicator_loader(self, negation_file_path):
        negation_indicators = set()
        with open(negation_file_path, 'r') as f:
            for line in f:
                negation_indicators.add(line.strip('\n '))
        return negation_indicators

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
                tokens[i] = 'posnum'
            elif re.search(pattern_neg, tokens[i]):
                tokens[i] = 'negnum'
            i += 1
        return tokens

    def post_num_and_negation_strip(self, tokens):
        return [token.strip(',.!;:?()#&- ') for token in tokens]

    def count_all_caps(self, original_text):
        pattern = re.compile(r"\s[A-Z]+\s")
        all_caps_tokens = re.findall(pattern, original_text)
        long_caps_tokens = [token for token in all_caps_tokens if len(token.strip("\t\n-+&$% ")) > 1]
        count = len(long_caps_tokens)
        return count

    def count_elongated(self, tokens):
        pattern = re.compile(r"(.)\1{2}")
        return len([token for token in tokens if pattern.search(token)])

    def count_negated(self, tokens):
        return len([token for token in tokens if "neg_" in token])

    def count_emoticons(self, emoticon_list):
        return len(emoticon_list)

    def find_bi_grams(self, tokens):
        bigrm = [" ".join(bigram_tokens) for bigram_tokens in bigrams(tokens)]
        tokens += bigrm
        return tokens

    def feature_generator(self, tweet_data):
        feature_dict = {}
        tokens = tweet_data['processed_tokens']
        original_text = tweet_data['original_text']
        emoticon_list = tweet_data['emoticon_list']

        tokens = self.num_handling(tokens)
        tokens = self.negation_handling(tokens)
        stripped_tokens = self.post_num_and_negation_strip(tokens)
        feature_dict['all_caps_count'] = self.count_all_caps(original_text)
        feature_dict['elongated_count'] = self.count_elongated(stripped_tokens)
        feature_dict['negated_count'] = self.count_negated(stripped_tokens)
        feature_dict['emoticon_count'] = self.count_emoticons(emoticon_list)
        n_grams = self.find_bi_grams(stripped_tokens)
        
        tweet_data['n-grams'] = n_grams
        tweet_data['engineered features'] = feature_dict

        return tweet_data

    def feature_updating(self):
        count = 0
        with open(self.input_path, 'r') as f_in, open(self.output_path, 'w') as f_out:
            for line in f_in:
                tweet_data = json.loads(line)
                updated_tweet_data = self.feature_generator(tweet_data)

                # write updated tweet_data to f_out
                f_out.write(json.dumps(updated_tweet_data) + "\n")
                count += 1
                if count % 1000 == 0:    
                    print(count)
        return 

class features2Instance:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.word2index = self.get_word2index()

    def get_word2index(self):
        word2index = {}
        for i, word in enumerate(self.vocabulary):
            word2index[word] = i
        
        return word2index 

    def get_dense_vec(self, tweet_tokens, feature_dictionary):
        dense_tweet_vector = np.zeros(len(self.vocabulary))
        engineered_feature_values = np.fromiter(feature_dictionary.values(), dtype=float)
        
        for token in tweet_tokens:
            # OOV tokens will not be considered in the test data
            token_index = self.word2index.get(token, None)
            if token_index != None:
                dense_tweet_vector[token_index] += 1
        
        return np.concatenate((dense_tweet_vector, engineered_feature_values))

    def get_sparse_matrix(self, dense_vec):
        return sparse.coo_matrix(dense_vec)

    def get_sparse_matrix_from_tweet(self, tweet_tokens, feature_dictionary):
        dense_vec = self.get_dense_vec(tweet_tokens, feature_dictionary)
        return self.get_sparse_matrix(dense_vec)

"""-----------------Below needs to be added to class of instance2tweet-----------------"""

def read_line(line):
    tweet_data = json.loads(line)
    tokens = tweet_data['n-grams']
    feature_dictionary = tweet_data['engineered features']
    return tokens, feature_dictionary

def append_sparse_matrix(new_sparse_vec, sparse_matrix):
    vector_to_stack = new_sparse_vec.T
    sparse_matrix = sparse.hstack((sparse_matrix, vector_to_stack))
    return sparse_matrix

def build_sparse_matrix(tweet_path, vocabulary):
    tweet2instance = features2Instance(vocabulary)
    with open(tweet_path, 'r') as f:
        
        # read the first line
        line = f.readline()
        tokens, feature_dictionary = read_line(line)
        sparse_matrix = tweet2instance.get_sparse_matrix_from_tweet(tokens, feature_dictionary).T
        count = 0
        
        # skip the first tweet
        while True:
            line = f.readline()
            if not line:
                break
            tokens, feature_dictionary = read_line(line)
            print(tokens)
            new_sparse_matrix = tweet2instance.get_sparse_matrix_from_tweet(tokens, feature_dictionary)
            print('catch 1')
            sparse_matrix = append_sparse_matrix(
                                new_sparse_matrix, 
                                sparse_matrix
            )
            print('catch 2')
            count += 1
            if count % 100 == 0:   
                print(count)

    return sparse_matrix
    
def string_compiler(sparse_indices):
    string_to_write = ""
    for index, value in sparse_indices.items():
        string_to_write += f"{int(index)}:{value} "
    return string_to_write

def write_data_set(output_path, sparse_matrix):
    with open(output_path, 'w') as f:
        for i in range(sparse_matrix.shape[1]):
            
            # TODO: there must be an off the shelf way to do this
            # so change to better code
            sparse_col = sparse_matrix.getcol(i)
            sparse_indices = sparse_col.nonzero()[0]
            # map the sparse index to its frequency in the tweet
            sparse_index2freq = {index: sparse_col.getrow(index).toarray()[0][0] for index in sparse_indices}
                        
            string_to_write = string_compiler(sparse_index2freq).strip(" ")
            f.write(string_to_write + "\n")

            if i % 100 == 0:
                print(i)

"""-----------------Above needs to be added to class of instance2tweet-----------------"""

def directory_builder(ticker):
    """
    takes a ticker and builds all the directories
    required for the labelling pipeline of the provided ticker.

    params:
        ticker: string
    output:
        None
    """
    ticker_string = ticker.strip('$')
    ticker_folder_path = f'../data/{ticker_string}/'
    paths = {}
    paths['raw'] = os.path.join(ticker_folder_path, 'raw_data')
    paths['duplicate'] = os.path.join(ticker_folder_path, 'duplicate_removed_data')
    paths['engineered'] = os.path.join(ticker_folder_path, 'engineered_data')
    paths['unseen'] = os.path.join(ticker_folder_path, 'unseen_instances')
    if not os.path.exists(ticker_folder_path):
        os.mkdir(ticker_folder_path)
        os.mkdir(paths['raw'])
        os.mkdir(paths['duplicate'])
        os.mkdir(paths['engineered'])
        os.mkdir(paths['unseen'])
    
    return paths

def daterange_2_dailyinterval(date_range):
    """
    takes a date range and returns a list of date
    intervals that can be used 

    params: 
        date_range: tuple
            begin and end date of the date range
            for which the tweets are to be collected
            and labelled
    output:
        daily_intervals: list
            list of tuples, where each tuple contains the 
            begin and end datetime object of the day
            for which the tweets are to be collected
    """
    begin_date = date_range[0]
    end_date = date_range[1]
    total_days = end_date - begin_date
    dates_in_range = [end_date - datetime.timedelta(days=x) for x in range(int(total_days.days))]
    daily_intervals = [(day, day + datetime.timedelta(days=1)) for day in dates_in_range]
    daily_intervals.reverse()

    return daily_intervals

def datetime_2_filename(start_date):
    """
    takes a datetime object and returns the string 
    that will be the filename for that day
    """
    date_string = start_date.strftime("%Y-%m-%d")
    file_name = date_string + ".txt"
    return file_name

def load_vocabulary(voc_path):
    with open(voc_path, 'r') as f:
        vocabulary = json.load(f)
    return vocabulary

def main(
    ticker_list, 
    date_range,
    client,
    similarity_threshold,
    high_frequency_threshold,
    negation_indicator_path,
    vocabulary_path
    ):
    for ticker in ticker_list:
        output_paths = directory_builder(ticker)
        daily_intervals = daterange_2_dailyinterval(date_range)
        for day in daily_intervals:
            day_start = day[0]
            day_end = day[1]
            daily_file_name = datetime_2_filename(day_start)

            # collect day's tweets and store in raw/filename
            raw_data_day_path = os.path.join(output_paths['raw'], daily_file_name)
            tweet_collector = TweetCollector(client, day_start, day_end, raw_data_day_path)
            tweet_collector.execute_company_search(ticker)

            # load raw/filename, remove duplicates and write to duplicate/filename
            duplicate_removed_day_path = os.path.join(output_paths['duplicate'], daily_file_name)
            duplicate_pre_processor = duplicatePreProcessor(
                                        raw_data_day_path,
                                        duplicate_removed_day_path,
                                        similarity_threshold, 
                                        high_frequency_threshold
            )
            duplicate_pre_processor.pre_process_collection()
            duplicate_pre_processor.set_word2index()
            duplicate_pre_processor.duplicate_filter()

            # load duplicate_filtered, engineer and write to engineered/filename
            engineered_day_path = os.path.join(output_paths['engineered'], daily_file_name)
            feature_engineering = featureEngineering(duplicate_removed_day_path, engineered_day_path, negation_indicator_path)
            feature_engineering.feature_updating()

            # turn the engineered features into instances and write instances
            instance_day_path = os.path.join(output_paths['unseen'], daily_file_name)
            vocabulary = load_vocabulary(vocabulary_path)
            sparse_total_matrix = build_sparse_matrix(engineered_day_path, vocabulary)
            write_data_set(instance_day_path, sparse_total_matrix)


if __name__ == "__main__":
    NEGATION_INDICATOR_PATH = '../data/negation_ind.txt'
    VOCABULARY_PATH = '../data/total_train_vocabulary.txt'
    TICKER_LIST = ["AAPL"]
    START_TIME = datetime.datetime(2022, 7, 15, 0, 0, 0, 0, datetime.timezone.utc)
    END_TIME = datetime.datetime(2022, 7, 16, 0, 0, 0, 0, datetime.timezone.utc)
    DATE_RANGE = (START_TIME, END_TIME)
    BEARER = os.environ.get("BEARER")
    CLIENT = Twarc2(bearer_token=BEARER)
    HIGH_FREQ_THRESHOLD = 10
    SIMILARITY_THRESHOLD = 0.6

    main(
        TICKER_LIST,
        DATE_RANGE,
        CLIENT,
        SIMILARITY_THRESHOLD,
        HIGH_FREQ_THRESHOLD,
        NEGATION_INDICATOR_PATH,
        VOCABULARY_PATH
    )