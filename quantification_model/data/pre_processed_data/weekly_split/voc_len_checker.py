import json
with open('weekly_split_voc.txt', 'r') as f:
	voc = json.load(f)
	print(len(voc))
