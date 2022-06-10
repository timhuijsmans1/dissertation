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
        with open(os.path.join(path, f"page_{i}.txt"), 'w+') as f:
            for tweet in result:
                print(tweet["text"])
                f.write('%s\n' % json.dumps(tweet))

if __name__ == "__main__":
    # global vars
    BEARER = os.environ.get("BEARER")
    CLIENT = Twarc2(bearer_token=BEARER)
    QUERY = "($ATVI OR $ADBE OR $ADP OR $ABNB OR $ALGN OR $GOOGL OR $GOOG OR $AMZN OR $AMD OR $AEP OR $AMGN OR $ADI OR $ANSS OR $AAPL OR $AMAT OR $ASML OR $AZN OR $TEAM OR $ADSK OR $BIDU OR $BIIB OR $BKNG OR $AVGO OR $CDNS OR $CHTR OR $CTAS OR $CSCO OR $CTSH OR $CMCSA OR $CEG OR $CPRT OR $COST OR $CRWD OR $CSX OR $DDOG OR $DXCM OR $DOCU OR $DLTR OR $EBAY OR $EA OR $EXC OR $FAST OR $FISV OR $FTNT OR $GILD OR $HON OR $IDXX OR $ILMN OR $INTC OR $INTU OR $ISRG OR $JD OR $KDP OR $KLAC OR $KHC OR $LRCX OR $LCID OR $LULU OR $MAR OR $MRVL OR $MTCH OR $MELI OR $FB OR $MCHP OR $MU OR $MSFT OR $MRNA OR $MDLZ OR $MNST OR $NTES OR $NFLX OR $NVDA OR $NXPI OR $ORLY OR $OKTA OR $ODFL OR $PCAR OR $PANW OR $PAYX OR $PYPL OR $PEP OR $PDD OR $QCOM OR $REGN OR $ROST OR $SGEN OR $SIRI OR $SWKS OR $SPLK OR $SBUX OR $SNPS OR $TMUS OR $TSLA OR $TXN OR $VRSN OR $VRSK OR $VRTX OR $WBA OR $WDAY OR $XEL OR $ZM) (😀 OR 😃 OR 😄 OR 😁 OR 🙂 OR 😊 OR 🤩 OR 🤑 OR 🚀 OR 💎 OR 😡 OR 😤 OR 😟 OR 😰 OR 😨 OR 😖 OR 😩 OR 🤬 OR 😠 OR 💀) lang:en -is:retweet"
    START_TIME = datetime.datetime(2017, 11, 9, 0, 0, 0, 0, datetime.timezone.utc)
    END_TIME = datetime.datetime(2022, 5, 30, 0, 0, 0, 0, datetime.timezone.utc)
    OUTPUT_PATH = "./test_output"
    
    # execute program
    results = execute_search(CLIENT, QUERY, START_TIME, END_TIME)
    result_writer(OUTPUT_PATH, results)