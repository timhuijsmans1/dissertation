import os
import datetime
import json

from twarc import Twarc2, expansions

class TweetCollector():

    def __init__(self, 
                ticker_path, 
                emoticon_list, 
                client, 
                start_datetime, 
                end_datetime,
                output_path
        ):
        self.ticker_list = self.ticker_loader(ticker_path)
        self.emoticon_list = emoticon_list
        self.client = client
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.output_path = output_path

    def ticker_loader(self, path):
        ticker_list = []
        with open(path, 'r') as f:

            # read all company lines
            while True:
                line = f.readline()
                if not line:
                    break
                # extract ticker symbol from company line
                ticker = line.split(',')[1]
                ticker_list.append(ticker)

        return ticker_list
    
    def query_compiler(self):
        # compile lists of tickers and emoticons with OR bool
        ticker_query = "(" + " OR ".join(self.ticker_list) + ")"
        emoticon_query = "(" + " OR ".join(self.emoticon_list) + ")"

        # combine tickers, emoticons and query requirements
        self.query = ticker_query + " " + emoticon_query + " lang:en -is:retweet"

        return

    def execute_search(self):
        self.search_results = self.client.search_all(
                            query=self.query, 
                            start_time=self.start_datetime, 
                            end_time=self.end_datetime, 
                            max_results=100
        )

        return

    def result_filter(self, tweet):
        emoticon_bool = False
        ticker_bool = False
        
        tweet_text = tweet["text"]

        # check if the tweet indeed contains a nasdaq 100 ticker or an emoticon
        for ticker in self.ticker_list:
            if ticker in tweet_text:
                ticker_bool = True
                break
        for emoticon in self.emoticon_list:
            if emoticon in tweet_text:
                emoticon_bool = True
                break
        
        return ticker_bool * emoticon_bool

    def result_writer(self):
        good_tweets = 0
        false_tweets = 0
        with open(self.output_path, 'w+') as f:  
            for page in self.search_results:
                result = expansions.flatten(page)

                # check if tweets are valid and write valid tweets
                for tweet in result:
                    if self.result_filter(tweet) == True:
                        f.write('%s\n' % json.dumps(tweet))
                        good_tweets += 1
                    else:
                        false_tweets += 1
                print(datetime.datetime.now())
                print("good tweets: ", good_tweets)
                print("false tweets: ", false_tweets)
                print("sum: ", false_tweets + good_tweets)
                print('-' * 30)
        return
                    
    def execute(self):
        self.query_compiler()
        self.execute_search()
        self.result_writer()

        return


if __name__ == "__main__":
    # global vars
    NASDAQ_100_PATH = "./nasdaq_100_listings.csv"
    OUTPUT_PATH = "./collected_data/filtered_search_results"
    BEARER = os.environ.get("BEARER")
    CLIENT = Twarc2(bearer_token=BEARER)
    START_TIME = datetime.datetime(2017, 11, 9, 0, 0, 0, 0, datetime.timezone.utc)
    END_TIME = datetime.datetime(2022, 5, 30, 0, 0, 0, 0, datetime.timezone.utc)
    POS_EMOTICON_LIST = ["😀", "😃", "😄", "😁", "🙂", "😊", "🤩", "🤑", "🚀", "💎"]
    NEG_EMOTICON_LIST = ["😡", "😤", "😟", "😰", "😨", "😖", "😩", "🤬", "😠", "💀"]
    EMOTICON_LIST = POS_EMOTICON_LIST + NEG_EMOTICON_LIST

    # execute program
    tweet_collector = TweetCollector(
                        NASDAQ_100_PATH, 
                        EMOTICON_LIST,
                        CLIENT,
                        START_TIME,
                        END_TIME,
                        OUTPUT_PATH
    )
    tweet_collector.execute()
