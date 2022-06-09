import os
import datetime
import json

from twarc import Twarc2, expansions

def execute_search(client, query, start_time, end_time):
    search_results = client.search_all(
                            query=query, 
                            start_time=start_time, 
                            end_time=end_time, 
                            max_results=100
    )

    return search_results

def result_writer(path, results):
    for i, page in enumerate(results):
        result = expansions.flatten(page)
        print(f"this page has {len(result)} results")
        # print(result)
        with open(os.path.join(path, f"page_{i}"), 'w+') as f:
            for tweet in result:
                f.write('%s\n' % json.dumps(tweet))

if __name__ == "__main__":
    # global vars
    BEARER = os.environ.get("BEARER")
    CLIENT = Twarc2(bearer_token=BEARER)
    QUERY = "(AAPL OR MSFT) (😀 OR 😃 OR 😄 OR 😁 OR 🙂 OR 😊 OR 🤩 OR 🤑) lang:en -is:retweet"
    START_TIME = datetime.datetime(2017, 11, 9, 0, 0, 0, 0, datetime.timezone.utc)
    END_TIME = datetime.datetime(2022, 5, 30, 0, 0, 0, 0, datetime.timezone.utc)
    OUTPUT_PATH = "./test_output"
    
    # execute program
    results = execute_search(CLIENT, QUERY, START_TIME, END_TIME)
    result_writer(OUTPUT_PATH, results)