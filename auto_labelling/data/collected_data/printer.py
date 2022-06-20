import json

with open('nasdaq_100_filtered_search_results_cashtag.txt') as f:
	while True:	
		tweet = f.readline()
		if not tweet:
			break
		text = json.loads(tweet)['text']
		print(text)
