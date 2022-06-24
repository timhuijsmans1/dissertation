import json
import os
import regex
import re
import sys
import emoji as emoji_lib
import numpy as np
import line_profiler
import atexit

from scipy import sparse
from scipy.spatial.distance import cosine

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

class preProcessor:

    def __init__(
        self, 
        raw_data_path, 
        pre_processed_path, 
        output_path, 
        collection_data_path,
        threshold 
        ):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.pre_processed_path = pre_processed_path
        self.collection_data_path = collection_data_path
        self.threshold = threshold

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
        # replace all multiple spaces in tweet by single spaces
        tweet_text = re.sub(r'\s+', ' ', tweet_text)
        # remove all web links from tweet
        tweet_text = re.sub(r'http\S+', '', tweet_text)
        # remove all usernames from tweet  
        tweet_text = re.sub(r'@\S+', '', tweet_text)
        # replace all hyphens by a space
        tweet_text = tweet_text.replace('-', ' ')
        # replace all underscores by a space
        tweet_text = tweet_text.replace('_', ' ')

        return tweet_text

    def tokenisation(self, tweet_text):

        # split tweet on spaces
        tweet_tokens = tweet_text.split(' ')

        # strip all punctuation of tokens
        tweet_tokens = [token.strip(',.!;:?-()# ') for token in tweet_tokens]

        # remove all tokens not containing any alphanumeric characters
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

    def pre_processing(self):
        with open(self.raw_data_path, 'r') as f_in:
            with open(self.pre_processed_path, 'w') as f_out:
                self.tweet_count = 0
                while True:
                    line = f_in.readline()
                    if not line:
                        break
                    else:
                        tweet_data = json.loads(line)
                        tweet_text = tweet_data['text']

                        # run pre_processing on individual tweets
                        pre_processed_tweet = self.tweet_pre_processing(tweet_text)
                        pre_processed_text = pre_processed_tweet['tweet_tokens']
                        self.vocabulary |= set(pre_processed_text)
                        
                        # write tweet data to the pre_processed_data file
                        tweet_data['pre_processed_text'] = pre_processed_tweet
                        f_out.write(json.dumps(tweet_data) + '\n')

                        self.tweet_count += 1
                        
                        # print tweet count every 1000 tweets
                        if self.tweet_count % 1000 == 0:
                            print('Processed {} tweets'.format(self.tweet_count))

        # write collection data to disk
        collection_data = {'tweet_count': self.tweet_count,
                            'vocabulary': list(self.vocabulary)}
        with open(self.collection_data_path, 'w+') as f:
            json.dump(collection_data, f)
        return

    def load_collection_data(self):
        with open(self.collection_data_path, 'r') as f:
            collection_data = json.load(f)
            self.tweet_count = collection_data['tweet_count']
            self.vocabulary = collection_data['vocabulary']
        return

    def get_word2index(self):
        self.word2index = {}
        for i, word in enumerate(self.vocabulary):
            self.word2index[word] = i

        # swap all keys and values of word2index
        self.index2word = {v: k for k, v in self.word2index.items()}
        
        return

    def dense_vector_from_tokens(self, tokens):
        # build sparse tweet vector
        dense_tweet_vector = np.zeros(len(self.vocabulary))
        for token in tokens:
            token_index = self.word2index[token]
            dense_tweet_vector[token_index] += 1
        return dense_tweet_vector

    @profile
    def cosine_similarity_check(self, threshold, vector_1, vector_2):
        """
        calculate cosine similarity between vector 1 and 2: return
        True if higher than threshold and False if lower than threshold
        """
        cosine_similarity = np.dot(vector_1, vector_2) \
            / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))

        # cosine_similarity = cosine(vector_1, vector_2)
        return cosine_similarity >= threshold
    

    def duplicate_check(self, sparse_vector_list, vector_to_check):
        # assume that the tweet is no duplicate until tested
        new_tweet = True
        for vector in sparse_vector_list:
            # break the loop if the tweet vector is a duplicate
            dense_existing_vector = vector.todense()[0]

            if self.cosine_similarity_check(
                        self.threshold, 
                        dense_existing_vector,
                        vector_to_check
                ):
                new_tweet = False
                break

        return new_tweet
                            
    def duplicate_filter(self):
        sparse_vectors = []
        with open(self.pre_processed_path, 'r') as f_in:
            with open(self.output_path, 'w') as f_out:
                count = 0
                while True:
                    line = f_in.readline()
                    if not line:
                        break
                    else:
                        tweet_data = json.loads(line)
                        pre_processed_text = (
                            tweet_data['pre_processed_text']['tweet_tokens']
                        )
                        dense_new_vector = self.dense_vector_from_tokens(
                                                            pre_processed_text
                        )

                        # check if duplicate or similar, always add the first
                        # vector to the list of sparse vectors
                        if sparse_vectors:
                            new_tweet = self.duplicate_check(
                                                sparse_vectors, 
                                                dense_new_vector
                            )
                            # if the tweet vector is not similar, add it to the list
                            # and write the corresponding tweet data to the output file
                            if new_tweet:
                                sparse_vectors.append(sparse.coo_array(dense_new_vector))
                                f_out.write(line)
                        else:
                            sparse_vectors.append(sparse.coo_array(dense_new_vector))
                            f_out.write(line)

                        count += 1
                        if count % 1000 == 0:
                            sparse_vectors = []
                        print(count)
                    
        return                        

                        
if __name__ == "__main__":
    TEST_DATA_PATH = "../data/collected_data/test_raw.txt"
    FULL_DATA_PATH = "../data/collected_data/filtered_search_results_06-20-2022_15;50;14.txt"
    PREPROCESSED_PATH = "../data/preprocessed_data/preprocessed.txt"
    COLLECTION_DATA_PATH = "../data/preprocessed_data/collection_data.txt"
    OUTPUT_PATH = "../data/preprocessed_data/final_data.txt"
    COSINE_SIM_THRESHOLD = 0.6
    
    pre_processor = preProcessor(
                        FULL_DATA_PATH, 
                        PREPROCESSED_PATH, 
                        OUTPUT_PATH, 
                        COLLECTION_DATA_PATH,
                        COSINE_SIM_THRESHOLD
    )
    # make sure to clear the final data file before running
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    # if the path does not exist, we need to pre-process all tweets and 
    # generate the pre-processed vocabulary first.
    if not os.path.exists(PREPROCESSED_PATH):  
        pre_processor.pre_processing()
    # the collection data consists of the number of tweets and the vocabulary
    else:
        pre_processor.load_collection_data()
    pre_processor.get_word2index()
    pre_processor.duplicate_filter()